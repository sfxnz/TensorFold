"""Concurrent requests over 4 lanes on one rank: every stream equals its solo run on the legacy engine (tokens, each
verify window's logits bits, drafted, accepted and rounds) with 2, 3 and 4 streams, greedy and keyed, fixed and
confidence policies, a serial lane, graphs on and off, proposals alone and in one block (made early, late or lazily,
kept rows absorbed alone or in one pass); prompts admitted mid-decode
fill beside the rounds at every decode share, in small spans, paused and resumed; clients that leave free their lanes;
and a Scheduler from 4 threads, with a background stream that yields and replays, equals solo too.

Prompts cross the tiny config's transitions (16 visible entries, 32-entry candidate pools, the 128-slot ring).
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_tiny
from dsv41_ref_weights import RefWeights

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import BLOCK, MAX_ROWS, PREFILL_ROWS, dspark, loader, multi_fill, proposals
from tensorfold.families.deepseek_v41.cuda import decode as D
from tensorfold.families.deepseek_v41.cuda import multi as M
from tensorfold.families.deepseek_v41.cuda import prefill as P
from tensorfold.families.deepseek_v41.cuda.buffers import Buffers
from tensorfold.families.deepseek_v41.cuda.graphs import LaneGraphs
from tensorfold.families.deepseek_v41.cuda.lanes import Ahead, Lanes

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
CAP = 1024
S = 4                                   # lanes
CONF = 0.3
POLICIES = {"fixed": (3, None), "confidence": (BLOCK, CONF)}
GRID = [(t, k, p) for t in (0.0, 1.0) for k in (0, 20) for p in (0.95, 1.0)]
LENGTHS = (5, 100, 600, 300)            # the 16/32-entry transitions, the ring's wrap, full lists
COUNTS = (40, 23, 31, 17)               # lanes finish in different rounds
SHARES = (0.0, 0.25, 0.5, 4.0)


class Req(NamedTuple):
    seed: int
    n: int
    sampling: Sampling | None
    count: int
    draft: bool = True
    stop_eos: bool = True
    background: bool = False


def _ids(seed: int, n: int, vocab: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, vocab, (n,), generator=g).tolist()


def _bits(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _sampling(i: int, seed: int) -> Sampling | None:
    t, k, p = GRID[i % len(GRID)]
    return Sampling(seed=seed, temperature=t, top_k=k, top_p=p) if t > 0 else None


@pytest.fixture(scope="module")
def ref(tiny_dir):
    return RefWeights(tiny_dir)


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    return loader.load(tiny_dir, Config.read(tiny_dir), 0, 1, None, capacity=CAP)


@pytest.fixture(scope="module")
def oracle(tiny, ref):
    return D.Engine(tiny, CAP, graphs=True, hasher=ref.hasher, reader=ref.reader)


def _rig(w, ref, rows: int = min(PREFILL_ROWS, CAP)) -> tuple:
    """S lane Engines on one stack's views, sharing ``rows``-row prompt buffers and the shared forward's."""

    cfg, dev = w.cfg, w.device
    lanes = Lanes(cfg, S, CAP, dev)
    pbuf = Buffers(cfg, w.world, rows, CAP, prefill=True, device=dev)
    mbuf = Buffers(cfg, w.world, MAX_ROWS * S, CAP, device=dev, lanes=S)
    return lanes, mbuf, pbuf, [D.Engine(w, CAP, hasher=ref.hasher, reader=ref.reader, st=lanes.view(k), pbuf=pbuf,
                                        dbuf=mbuf, dwork=dspark.Work(cfg, w.world, dev), ahead=Ahead(cfg, w.world))
                               for k in range(S)]


@pytest.fixture(scope="module")
def rig(tiny, ref):
    return _rig(tiny, ref)


@pytest.fixture(scope="module")
def graphs(tiny, rig):
    lanes, mbuf, _, engines = rig
    lg = LaneGraphs(tiny, lanes, mbuf, engines)
    assert lg.warm() == 22 * S
    return lg


@pytest.fixture(scope="module")
def batch(tiny):
    k = proposals.Batch(tiny.cfg, tiny.world, S, tiny.device, absorb=True)
    k.absorbs = False                   # its absorb tables are used where a test turns them on
    return k


@pytest.fixture(scope="module")
def batched_graphs(tiny, rig, batch):
    lanes, mbuf, _, engines = rig
    lg = LaneGraphs(tiny, lanes, mbuf, engines, batch)
    assert lg.warm() == 22 * S + BLOCK * sum(L + 1 for L in range(2, S + 1))     # by (lanes, keyed of them, d)
    return lg


@pytest.fixture(scope="module")
def late_graphs(tiny, rig):
    """Lane graphs without a block: build_batch captures the block's into a pool of their own."""

    lanes, mbuf, _, engines = rig
    lg = LaneGraphs(tiny, lanes, mbuf, engines)
    assert lg.warm() == 22 * S
    return lg


def _decoder(w, rig, policy, graphs=None, share: float = 0.5, batch=None) -> M.LaneDecoder:
    lanes, mbuf, pbuf, engines = rig
    return M.LaneDecoder(w, lanes, engines, mbuf, pbuf, policy, (w.cfg.eos_token_id,), graphs, share, batch=batch)


_SOLO: dict = {}


def _solo(e: D.Engine, req: Req, policy) -> dict:
    """The legacy engine's run of ``req``: its tokens, each verify window's logits rows, its draft stats."""

    key = (req._replace(background=False), policy)
    if key in _SOLO:
        return _SOLO[key]
    windows, forward = [], e.forward

    def fwd(tokens):
        out = forward(tokens)
        windows.append([_bits(row) for row in out])
        return out

    prompt = _ids(req.seed, req.n, e.w.cfg.vocab_size)
    e.forward = fwd
    try:
        first = P.prefill(e, prompt, req.sampling)
        if req.draft and policy[0]:
            res = D.dspark_decode(e, first, req.count, req.sampling, drafts=policy[0], confidence=policy[1],
                                  stop_eos=req.stop_eos)
        else:
            res = D.serial_decode(e, first, req.count, req.sampling, stop_eos=req.stop_eos)
    finally:
        del e.forward
    _SOLO[key] = {"tokens": res.tokens, "windows": windows, "stats": (res.drafted, res.accepted, res.rounds)}
    return _SOLO[key]


class Run:
    """Streams through ``dec`` as the Scheduler drives it: each admitted once its condition holds and a lane is free,
    then a round and ``finish`` a step; every stream's emitted tokens and verify windows are kept."""

    def __init__(self, dec: M.LaneDecoder) -> None:
        self.dec, self.got, self.windows, self.spans, self.steps = dec, {}, {}, {}, 0
        dec._draws = self._recorder(dec._draws)

    def _recorder(self, draws):
        def rec(live):
            for s in live:
                self.windows.setdefault(id(s), []).append([_bits(row) for row in self.dec.mbuf.logits[s.seg0:s.seg0 + s.R]])
            return draws(live)
        return rec

    def stream(self, req: Req, stop=None) -> Stream:
        """``req`` as a Stream; ``stop(tokens so far)`` True makes its client leave."""

        s = Stream(_ids(req.seed, req.n, self.dec.w.cfg.vocab_size), req.count, req.sampling, draft=req.draft,
                   stop_eos=req.stop_eos, background=req.background)
        got = self.got[id(s)] = []
        s.emit = lambda new: (got.extend(new), stop is not None and stop(got))[1]
        return s

    def go(self, arrivals: list, decode_seen=None) -> None:
        waiting = list(arrivals)
        while waiting or self.dec.live():
            for cond, s in list(waiting):
                if self.dec.live() < S and cond(self):
                    self.dec.admit(s)
                    self.spans[id(s)] = {z - a for a, z in s.fill.spans}
                    waiting.remove((cond, s))
            done = self.dec.round()
            self.dec.finish(done)
            self.steps += 1
            check(self.dec)
            assert self.steps < 5000, "the decoder made no progress"

    def close(self) -> None:
        del self.dec._draws


def check(dec: M.LaneDecoder) -> None:
    """M4: decoding streams only in ``streams``, filling ones only in ``filling``, each context prompt + reply."""

    assert dec._invariants()
    filling = {id(s) for s in dec.filling}
    for s in dec.streams.values():
        assert id(s) not in filling and s.context == [*s.prompt, *s.out]


def _same(run: Run, s: Stream, want: dict, what: str, *, cut: bool = False) -> None:
    """``s`` equals its solo run: every token and window, and the draft stats (a stream cut short: its prefix)."""

    got, windows = run.got[id(s)], run.windows.get(id(s), [])
    if cut:
        assert got == want["tokens"][:len(got)] and windows == want["windows"][:len(windows)], what
        return
    assert got == want["tokens"], f"{what}: tokens"
    assert windows == want["windows"], f"{what}: a verify window's logits bits"
    assert (s.drafted, s.accepted, s.rounds) == want["stats"], f"{what}: drafted, accepted, rounds"


def _always(run: Run) -> bool:
    return True


# -- M1: together equals solo ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("graphed", [True, False], ids=["graphs", "eager"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("c", [2, 3, 4])
def test_streams_together_equal_their_solo_runs(tiny, oracle, rig, graphs, c, policy, graphed):
    """c streams admitted at once: greedy and keyed, one serial (``"draft": false``), one with ``stop_eos`` off,
    different counts; each equals its solo run bit for bit, its draft stats included."""

    shift = c + 3 * list(POLICIES).index(policy)
    reqs = [Req(10 * c + j, LENGTHS[j], _sampling(shift + j, j), COUNTS[j], draft=j != c - 1, stop_eos=j != 0)
            for j in range(c)]
    want = [_solo(oracle, r, POLICIES[policy]) for r in reqs]
    assert all(w["stats"][0] > 0 for r, w in zip(reqs, want) if r.draft), "every drafting stream drafts alone"
    run = Run(_decoder(tiny, rig, POLICIES[policy], graphs if graphed else None))
    try:
        streams = [run.stream(r) for r in reqs]
        run.go([(_always, s) for s in streams])
    finally:
        run.close()
    for j, (s, w) in enumerate(zip(streams, want)):
        _same(run, s, w, f"stream {j} of {c}, {reqs[j]}")
    assert not run.dec.live() and run.dec._free() == list(range(S))


@pytest.mark.parametrize("graphed", [True, False], ids=["graphs", "eager"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("c", [2, 3, 4])
def test_proposals_in_one_block_equal_the_solo_runs(tiny, oracle, rig, batch, batched_graphs, monkeypatch, c, policy,
                                                     graphed):
    """c streams, greedy and keyed mixed, all drafting but one at 4: their lanes propose in one block, and each
    stream equals its solo run bit for bit, its draft stats included."""

    reqs = [Req(20 * c + j, LENGTHS[j], _sampling(3 * j + c, j), COUNTS[j], draft=c < 4 or j != 3, stop_eos=j != 0)
            for j in range(c)]
    want = [_solo(oracle, r, POLICIES[policy]) for r in reqs]
    blocks, propose = [], proposals.propose
    monkeypatch.setattr(proposals, "propose", lambda k, wanted, *a, **kw: (blocks.append(len(wanted)),
                                                                           propose(k, wanted, *a, **kw))[1])
    run = Run(_decoder(tiny, rig, POLICIES[policy], batched_graphs if graphed else None, batch=batch))
    try:
        streams = [run.stream(r) for r in reqs]
        run.go([(_always, s) for s in streams])
    finally:
        run.close()
    assert max(blocks) == sum(r.draft for r in reqs), "every drafting lane proposed in one block"
    for j, (s, w) in enumerate(zip(streams, want)):
        _same(run, s, w, f"stream {j} of {c} proposing together, {reqs[j]}")
    assert not run.dec.live() and run.dec._free() == list(range(S))


@pytest.mark.parametrize("absorb", [False, True], ids=["absorb alone", "absorb together"])
@pytest.mark.parametrize("graphed", [True, False], ids=["graphs", "eager"])
@pytest.mark.parametrize("when", proposals.SETUPS)
@pytest.mark.parametrize("c", [2, 4])
def test_blocks_made_early_late_or_lazily_equal_the_solo_runs(tiny, oracle, rig, batch, batched_graphs, late_graphs,
                                                               monkeypatch, c, when, graphed, absorb):
    """c drafting streams, greedy and keyed mixed, the block's scratch and graphs made before the lane graphs, after
    them or at the first round with two drafting lanes, and the kept rows absorbed alone or in one pass: each stream
    equals its solo run; a lazy block is built once, on a live lane's rows."""

    policy = POLICIES["confidence"]
    reqs = [Req(40 * c + j, LENGTHS[j], _sampling(5 * j + c, j), COUNTS[j]) for j in range(c)]
    want = [_solo(oracle, r, policy) for r in reqs]
    blocks, together, propose, absorb_all = [], [], proposals.propose, proposals.absorb
    monkeypatch.setattr(proposals, "propose", lambda k, wanted, *a, **kw: (blocks.append(len(wanted)),
                                                                           propose(k, wanted, *a, **kw))[1])
    monkeypatch.setattr(proposals, "absorb", lambda w, lanes, b, k, kept: (together.append(len(kept)),
                                                                           absorb_all(w, lanes, b, k, kept))[1])
    if when == "early":
        monkeypatch.setattr(batch, "absorbs", absorb)
        dec = _decoder(tiny, rig, policy, batched_graphs if graphed else None, batch=batch)
    else:
        dec = _decoder(tiny, rig, policy, late_graphs if graphed else None)
    built, build = [], dec.build_batch
    dec.build_batch = lambda a, lane=None: (built.append(lane), build(a, lane))[1]
    if when == "late":
        dec.build_batch(absorb)
        assert dec.batch is not None and dec.batch.absorbs == absorb
        if graphed:
            assert len(late_graphs.batched) == BLOCK * sum(L + 1 for L in range(2, S + 1))
            assert late_graphs.batch_pool != late_graphs.pool, "the block's graphs in a pool of their own"
    elif when == "lazy":
        dec.defer, dec.defer_absorb = True, absorb
    run = Run(dec)
    try:
        streams = [run.stream(r) for r in reqs]
        run.go([(_always, s) for s in streams])
    finally:
        run.close()
    assert max(blocks) == c, "every drafting lane proposed in one block"
    assert bool(together) == absorb and all(n > 1 for n in together)
    assert len(built) == (when != "early") and (when != "late" or built == [None])
    assert when != "lazy" or (built[0] in range(S) and not dec.defer), "built once, on a live lane's rows"
    for j, (s, w) in enumerate(zip(streams, want)):
        _same(run, s, w, f"stream {j} of {c}, block made {when}, absorb together {absorb}, {reqs[j]}")
    assert not run.dec.live() and run.dec._free() == list(range(S))


def test_a_lone_stream_never_builds_a_lazy_block(tiny, oracle, rig, late_graphs):
    req, policy = Req(77, LENGTHS[1], None, COUNTS[1]), POLICIES["confidence"]
    want = _solo(oracle, req, policy)
    dec = _decoder(tiny, rig, policy, late_graphs)
    dec.defer = True
    run = Run(dec)
    try:
        s = run.stream(req)
        run.go([(_always, s)])
    finally:
        run.close()
    _same(run, s, want, "a lone stream with the block deferred")
    assert dec.batch is None and dec.defer


def test_the_decoder_refuses_what_it_cannot_serve(tiny, rig):
    lanes, mbuf, pbuf, engines = rig
    with pytest.raises(ValueError, match="kept"):
        M.LaneDecoder(tiny, lanes, engines, mbuf, pbuf, (3, None), (1,), None, 0.5, kept=object())
    with pytest.raises(ValueError, match="engines"):
        M.LaneDecoder(tiny, lanes, engines[::-1], mbuf, pbuf, (3, None), (1,), None, 0.5)
    with pytest.raises(ValueError, match="drafts"):
        M.LaneDecoder(tiny, lanes, engines, mbuf, pbuf, (BLOCK + 1, None), (1,), None, 0.5)
    dec = _decoder(tiny, rig, (3, None))
    for prompt, match in (([], "empty"), ([3] * CAP, "positions"), ([tiny.cfg.vocab_size], "token ids")):
        with pytest.raises(ValueError, match=match):
            dec.admit(Stream(prompt, 4))
    s = Stream([3] * 10, 10 * CAP)
    dec.admit(s)
    assert s.count == CAP - 10 and s.lane == 0 and dec.filling == [s] and dec.live() == 1
    assert dec.drop() == [s] and not dec.live() and not dec.mbuf.split["hist"].any()


# -- M2: prompts admitted mid-decode ---------------------------------------------------------------------------------

@pytest.mark.parametrize("rows", [7, 129])
@pytest.mark.parametrize("share", SHARES)
def test_prompts_admitted_mid_decode_equal_their_solo_runs(tiny, ref, oracle, monkeypatch, share, rows):
    """Two short prompts and a long one admitted with nothing decoding (full spans, several a step), one admitted once
    two decode (BUSY_ROWS spans, filling beside the long one, which goes first once overdue, so each is paused with its
    next span's read in flight), and one admitted into the first lane freed (three decode at share 0); each equals its
    solo run."""

    busy = rows // 2
    monkeypatch.setattr(multi_fill, "BUSY_ROWS", busy)
    pauses, steps, pause, step = [], [], P.Fill.pause, P.Fill.step
    monkeypatch.setattr(P.Fill, "pause", lambda f: (pauses.append(f.reads.pending is not None), pause(f))[1])
    monkeypatch.setattr(P.Fill, "step", lambda f: (steps.append(f), step(f))[1])
    policy = POLICIES["confidence"]
    a1, a2 = Req(1, 3 * rows + 5, _sampling(1, 1), 10), Req(2, 3 * rows + 9, _sampling(2, 2), 300)
    x, y = Req(3, 6 * rows + 3, _sampling(3, 3), 40), Req(4, 10 * busy + 1, _sampling(4, 4), 40, draft=False)
    z = Req(5, 2 * rows + 1, _sampling(5, 5), 30, stop_eos=False)
    reqs = [a1, a2, x, y, z]
    want = [_solo(oracle, r, policy) for r in reqs]
    steps.clear()                                       # the solo prefills' own
    dec = _decoder(tiny, _rig(tiny, ref, rows), policy, share=share)
    decodes, real, decoding = [], dec._decode, []
    dec._decode = lambda: (decodes.append(len(dec.filling)), real())[1]
    run = Run(dec)
    try:
        sa1, sa2, sx, sy, sz = streams = [run.stream(r) for r in reqs]

        def joined(r: Run) -> bool:
            if not (sx.out and sy.out):
                return False
            decoding.append(sum(not s.done for s in dec.streams.values()))
            return True

        run.go([(_always, sa1), (_always, sa2), (_always, sx), (lambda r: bool(sa2.out), sy), (joined, sz)])
    finally:
        run.close()
    for j, (s, w) in enumerate(zip(streams, want)):
        _same(run, s, w, f"stream {j} at share {share}, {rows}-row spans")
    assert [max(run.spans[id(s)]) for s in streams] == [rows, rows, rows, busy, busy]
    assert any(pauses), "a fill was paused with its next span's read in flight"
    assert len({id(f) for f in steps[:multi_fill.FILL_SPANS]}) == 1, "nothing decoding: 4 spans of one prompt a step"
    assert decoding[-1] >= 2 and (share or decoding[-1] == 3), f"{decoding[-1]} lanes decoding at the last admission"
    if share == 0:
        assert not any(decodes), "share 0: no round while a prompt fills"


# -- M3: clients that leave, the Scheduler ---------------------------------------------------------------------------

def test_clients_that_leave_free_their_lanes_and_the_others_stay_exact(tiny, oracle, rig, graphs):
    policy = POLICIES["confidence"]
    reqs = [Req(31, 60, _sampling(5, 1), 40), Req(32, 120, _sampling(1, 2), 40), Req(33, 200, None, 40),
            Req(34, 90, _sampling(6, 4), 40)]
    later = Req(35, 400, _sampling(7, 5), 40)
    want = [_solo(oracle, r, policy) for r in [*reqs, later]]
    run = Run(_decoder(tiny, rig, policy, graphs))
    try:
        stops = [lambda got: True, lambda got: len(got) >= 12, lambda got: len(got) > 1, None]
        streams = [run.stream(r, stop) for r, stop in zip(reqs, stops)]
        last = run.stream(later)
        freed = lambda r: streams[0].done and streams[1].done and streams[2].done
        run.go([*((_always, s) for s in streams), (freed, last)])
    finally:
        run.close()
    sent = [len(run.got[id(s)]) for s in streams[:3]]
    assert sent[0] == 1 and 12 <= sent[1] < 12 + MAX_ROWS and 1 < sent[2] <= 1 + MAX_ROWS, f"clients left at {sent}"
    assert streams[0].rounds == 0 and all(s.done and s.error is None for s in streams)
    for j, s in enumerate(streams[:3]):
        _same(run, s, want[j], f"stream {j}, its client gone", cut=True)
    _same(run, streams[3], want[3], "the stream whose client stayed")
    assert last.lane == min(s.lane for s in streams[:3]), "a later request takes the lowest lane freed"
    _same(run, last, want[4], "a later request in a freed lane")


def _submit(sched: Scheduler, vocab: int, req: Req, out: dict, key, started: threading.Event | None = None) -> None:
    got = []

    def emit(new):
        got.extend(new)
        if started is not None and len(got) >= 5:
            started.set()

    stats = sched.submit(_ids(req.seed, req.n, vocab), req.count, req.sampling, req.draft, emit,
                         stop_eos=req.stop_eos, background=req.background)
    out[key] = (got, stats)


def test_a_scheduler_from_four_threads_equals_solo_and_a_background_stream_replays(tiny, oracle, rig, graphs):
    policy, vocab = POLICIES["confidence"], tiny.cfg.vocab_size
    dec = _decoder(tiny, rig, policy, graphs)
    sched = Scheduler(dec, max_streams=S)
    try:
        reqs = [Req(41 + j, LENGTHS[j], _sampling(j + 2, j), COUNTS[j], draft=j != 3) for j in range(4)]
        out: dict = {}
        threads = [threading.Thread(target=_submit, args=(sched, vocab, r, out, j)) for j, r in enumerate(reqs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(600)
        for j, r in enumerate(reqs):
            want = _solo(oracle, r, policy)
            got, stats = out[j]
            assert got == want["tokens"], f"request {j}: tokens"
            assert (stats["drafted"], stats["accepted"], stats["rounds"]) == want["stats"], f"request {j}: stats"
        assert not dec.live() and sched.yields == 0
        back = Req(50, 700, _sampling(5, 9), 150, background=True)       # refills in BUSY_ROWS spans
        fronts = [Req(51 + j, 100 + 50 * j, _sampling(j, 10 + j), 120) for j in range(3)]
        short = Req(60, 30, None, 8)
        started, out = threading.Event(), {}
        threads = [threading.Thread(target=_submit, args=(sched, vocab, back, out, "back", started))]
        threads += [threading.Thread(target=_submit, args=(sched, vocab, r, out, j)) for j, r in enumerate(fronts)]
        for t in threads:
            t.start()
        assert started.wait(600), "the background stream decodes"
        end = time.monotonic() + 60
        while dec.live() < S and time.monotonic() < end:
            time.sleep(0.001)
        _submit(sched, vocab, short, out, "short")
        for t in threads:
            t.join(600)
    finally:
        sched.close()
    assert sched.yields >= 1, "the background stream gave its lane to the waiting foreground request"
    for key, r in [("back", back), *enumerate(fronts), ("short", short)]:
        want = _solo(oracle, r, policy)
        got, stats = out[key]
        assert got == want["tokens"], f"{key}: tokens, each sent once"
        if key != "back":
            assert (stats["drafted"], stats["accepted"], stats["rounds"]) == want["stats"], f"{key}: stats"
    assert out["back"][1]["rounds"] > _solo(oracle, back, policy)["stats"][2], "the replay's rounds count too"


def test_stage_clocks_reach_the_stats_when_asked(tiny, oracle, rig, monkeypatch):
    monkeypatch.setenv(M.STAGES_ENV, "1")
    policy, req = POLICIES["fixed"], Req(70, 50, _sampling(3, 7), 12)
    run = Run(_decoder(tiny, rig, policy))
    try:
        s = run.stream(req)
        run.go([(_always, s)])
    finally:
        run.close()
    _same(run, s, _solo(oracle, req, policy), "a stream with stage clocks")
    stages = s.stats()["stages_ms"]
    assert set(stages) == {*D.HOST_PARTS, "device"} and stages["device"] > 0 and stages["sampling"] > 0
    monkeypatch.delenv(M.STAGES_ENV)
    run = Run(_decoder(tiny, rig, policy))
    try:
        s = run.stream(req)
        run.go([(_always, s)])
    finally:
        run.close()
    assert "stages_ms" not in s.stats()
