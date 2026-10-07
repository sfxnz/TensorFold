"""Kept prompt snapshots over 4 lanes on one rank: a prompt resumes in place, from another busy lane's live rows, or
from rows one lane saved and another loads, while other lanes decode; entries in a lane being refilled are evicted, a
fill's resumed snapshot stays pinned while another admission reserves, ``drop()`` settles every fill, and one
conversation's resends spread over lanes. Every stream equals its fresh solo run on the legacy engine (tokens, each
verify window's logits bits, drafted, accepted and rounds)."""

from __future__ import annotations

import hashlib
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_tiny
import numpy as np
from dsv41_ref_weights import RefWeights

from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import BLOCK, MAX_ROWS, PREFILL_ROWS, dspark, loader, snapshot
from tensorfold.families.deepseek_v41.cuda import decode as D
from tensorfold.families.deepseek_v41.cuda import multi as M
from tensorfold.families.deepseek_v41.cuda import prefill as P
from tensorfold.families.deepseek_v41.cuda.buffers import Buffers
from tensorfold.families.deepseek_v41.cuda.cache import Kept, entries_wanted, ids_digest
from tensorfold.families.deepseek_v41.cuda.graphs import LaneGraphs
from tensorfold.families.deepseek_v41.cuda.lanes import Ahead, Lanes

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
CAP = 1024
S = 4                                   # lanes
POLICY = (BLOCK, 0.3)                   # the confidence policy


class Req(NamedTuple):
    prompt: tuple[int, ...]
    sampling: Sampling | None
    count: int
    draft: bool = True


def _ids(seed: int, n: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, 1024, (n,), generator=g).tolist()


def _req(prompt, seed: int | None, count: int, draft: bool = True) -> Req:
    sampling = None if seed is None else Sampling(seed=seed, temperature=1.0, top_k=20 * (seed % 2), top_p=0.95)
    return Req(tuple(prompt), sampling, count, draft)


def _bits(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


@pytest.fixture(scope="module")
def ref(tiny_dir):
    return RefWeights(tiny_dir)


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    w = loader.load(tiny_dir, Config.read(tiny_dir), 0, 1, None, capacity=CAP)
    assert w.cfg.vocab_size == 1024
    return w


@pytest.fixture(scope="module")
def oracle(tiny, ref):
    return D.Engine(tiny, CAP, graphs=True, hasher=ref.hasher, reader=ref.reader)


@pytest.fixture(scope="module")
def rig(tiny, ref):
    cfg, dev = tiny.cfg, tiny.device
    lanes = Lanes(cfg, S, CAP, dev)
    pbuf = Buffers(cfg, tiny.world, min(PREFILL_ROWS, CAP), CAP, prefill=True, device=dev)
    mbuf = Buffers(cfg, tiny.world, MAX_ROWS * S, CAP, device=dev, lanes=S)
    return lanes, mbuf, pbuf, [D.Engine(tiny, CAP, hasher=ref.hasher, reader=ref.reader, st=lanes.view(k), pbuf=pbuf,
                                        dbuf=mbuf, dwork=dspark.Work(cfg, tiny.world, dev), ahead=Ahead(cfg, tiny.world))
                               for k in range(S)]


@pytest.fixture(scope="module")
def graphs(tiny, rig):
    lanes, mbuf, _, engines = rig
    lg = LaneGraphs(tiny, lanes, mbuf, engines)
    assert lg.warm() == 22 * S
    return lg


@pytest.fixture
def moves(monkeypatch, rig):
    """Every row move Kept makes, as (what, lane) or ("copy_rows", from lane, to lane)."""

    engines, log = rig[3], []

    def lane(x) -> int:
        return next(k for k, e in enumerate(engines) if x is e or x is e.st)

    def spied(name: str, states: int):
        real = getattr(snapshot, name)

        def spy(*args):
            log.append((name, *(lane(x) for x in args[:states])))
            return real(*args)
        return spy

    for name, states in (("save_rows", 1), ("load_rows", 1), ("copy_rows", 2)):
        monkeypatch.setattr(snapshot, name, spied(name, states))
    return log


def _kept(rig, entries: int = 16) -> Kept:
    e = rig[3][0]
    per = snapshot.state_bytes(e) + snapshot._span(e.st.row_views(CAP))[1]
    return Kept(rig[3], 16 * per, entries)


def _decoder(w, rig, kept: Kept, graphs=None) -> M.LaneDecoder:
    lanes, mbuf, pbuf, engines = rig
    return M.LaneDecoder(w, lanes, engines, mbuf, pbuf, POLICY, (w.cfg.eos_token_id,), graphs, 0.0, kept=kept)


_SOLO: dict = {}


def _solo(e: D.Engine, req: Req) -> dict:
    """The legacy engine's fresh run of ``req``: its tokens, each verify window's logits rows, its draft stats."""

    if req in _SOLO:
        return _SOLO[req]
    windows, forward = [], e.forward

    def fwd(tokens):
        out = forward(tokens)
        windows.append([_bits(row) for row in out])
        return out

    e.forward = fwd
    try:
        first = P.prefill(e, list(req.prompt), req.sampling)
        if req.draft:
            res = D.dspark_decode(e, first, req.count, req.sampling, drafts=POLICY[0], confidence=POLICY[1],
                                  stop_eos=False)
        else:
            res = D.serial_decode(e, first, req.count, req.sampling, stop_eos=False)
    finally:
        del e.forward
    _SOLO[req] = {"tokens": res.tokens, "windows": windows, "stats": (res.drafted, res.accepted, res.rounds)}
    return _SOLO[req]


class Run:
    """Streams through ``dec`` as the Scheduler drives it, each admitted once its condition holds, then a round and
    ``finish`` a step; every stream's emitted tokens and verify windows are kept. Streams run to their counts
    (``stop_eos`` False) and prompts fill before rounds (share 0), so which lanes are busy at each admission is fixed."""

    def __init__(self, dec: M.LaneDecoder) -> None:
        self.dec, self.got, self.windows, self.reqs = dec, {}, {}, {}
        for k, e in enumerate(dec.engines):
            e.sample = self._recorder(k, e.sample)

    def _recorder(self, k: int, sample):
        def rec(logits, positions, sampling):
            s = next(s for s in self.dec.streams.values() if s.lane == k and not s.done)
            self.windows.setdefault(id(s), []).append([_bits(row) for row in logits])
            return sample(logits, positions, sampling)
        return rec

    def stream(self, req: Req) -> Stream:
        s = Stream(list(req.prompt), req.count, req.sampling, draft=req.draft, stop_eos=False)
        got = self.got[id(s)] = []
        s.emit = got.extend
        self.reqs[id(s)] = req
        return s

    def go(self, arrivals: list, until=None) -> None:
        """Admit each (condition, stream) once its condition holds, until nothing is live or ``until(self)``."""

        waiting, steps = list(arrivals), 0
        while waiting or self.dec.live():
            for cond, s in list(waiting):
                if self.dec.live() < S and cond():
                    self.dec.admit(s)
                    waiting.remove((cond, s))
            if until is not None and until():
                return
            self.dec.finish(self.dec.round())
            assert self.dec._invariants()
            steps += 1
            assert steps < 5000, "the decoder made no progress"

    def same(self, oracle: D.Engine, s: Stream, what: str, cached: int = 0) -> None:
        """``s`` ended equal to its fresh solo run, having resumed from ``cached`` kept tokens."""

        want = _solo(oracle, self.reqs[id(s)])
        assert s.done and s.error is None, f"{what}: {s.error}"
        assert self.got[id(s)] == want["tokens"], f"{what}: tokens"
        assert self.windows.get(id(s), []) == want["windows"], f"{what}: a verify window's logits bits"
        assert (s.drafted, s.accepted, s.rounds) == want["stats"], f"{what}: drafted, accepted, rounds"
        assert s.cached == cached, f"{what}: resumed from {s.cached} tokens, not {cached}"

    def close(self) -> None:
        for e in self.dec.engines:
            del e.sample


def _accounted(kept: Kept) -> None:
    """The arena holds exactly the entries' spans and the reservations of fills in flight."""

    want = {x.state for x in kept.cache} | {x.rows for x in kept.cache if x.rows is not None}
    want |= {p.at for p in kept.pinned.values() if p.at is not None}
    assert set(kept.arena.spans) == want
    for x in kept.cache:
        assert x.digest == hashlib.sha256(np.asarray(x.snap.ids, dtype=np.int32).tobytes()).hexdigest()
        assert (x.lane is None) == (x.rows is not None), "an entry's rows are in one lane or saved"


def _then(*streams: Stream):
    """A condition: every one of ``streams`` has ended."""

    return lambda: all(s.done for s in streams)


def _decoding(*streams: Stream):
    """A condition: every one of ``streams`` decodes."""

    return lambda: all(s.out and not s.done for s in streams)


def _always() -> bool:
    return True


def test_kept_refuses_other_lanes_and_entries_default_past_the_lanes(tiny, rig):
    lanes, mbuf, pbuf, engines = rig
    with pytest.raises(ValueError, match="kept"):
        M.LaneDecoder(tiny, lanes, engines, mbuf, pbuf, POLICY, (1,), None, 0.5, kept=Kept(engines[:1], 1 << 20, 4))
    assert entries_wanted({}) == 8 and entries_wanted({}, lanes=S) == 8 + S
    assert entries_wanted({"TF_DSV41_CACHE_ENTRIES": "3"}, lanes=S) == 3
    assert ids_digest([1, 2, 3]) == hashlib.sha256(np.asarray([1, 2, 3], dtype=np.int32).tobytes()).hexdigest()
    kept = _kept(rig)
    kept.reserve(None, 2)
    with pytest.raises(RuntimeError, match="in flight"):
        kept.reserve(None, 2)
    kept.settle(2)
    assert not kept.pinned and not kept.arena.spans


def test_a_resend_resumes_in_place_while_another_lane_decodes(tiny, oracle, rig, moves):
    kept = _kept(rig)
    run = Run(_decoder(tiny, rig, kept))
    a = _req(_ids(1, 300), 1, 20)
    x = _req(_ids(2, 150), 2, 400, draft=False)
    try:
        sx, sa = run.stream(x), run.stream(a)
        run.go([(_always, sx), (_always, sa)], until=_then(sa))
        nxt = _req([*a.prompt, *run.got[id(sa)], *_ids(3, 40)], 3, 30)
        again = _req(a.prompt, 4, 25)
        sn, sg = run.stream(nxt), run.stream(again)
        run.go([(_decoding(sx), sn), (lambda: sn.done and not sx.done, sg)])
    finally:
        run.close()
    assert (sx.lane, sa.lane, sn.lane, sg.lane) == (0, 1, 1, 1)
    assert moves == [("save_rows", 1)], "the resend saves the next turn's rows, which it does not resume"
    for s, what, cached in ((sx, "the decoding lane", 0), (sa, "the first turn", 0),
                            (sn, "the next turn, in place", len(a.prompt) - 1),
                            (sg, "an identical resend, in place", len(a.prompt) - 1)):
        run.same(oracle, s, what, cached)
    _accounted(kept)


@pytest.mark.parametrize("graphed", [True, False], ids=["graphs", "eager"])
def test_prompts_resume_from_another_busy_lanes_live_rows(tiny, oracle, rig, graphs, moves, graphed):
    """F6: a stream resumes from a snapshot whose rows another stream still decodes over; its oracle runs fresh."""

    kept = _kept(rig)
    run = Run(_decoder(tiny, rig, kept, graphs if graphed else None))
    a = _req(_ids(11, 260), 5, 200)
    nxt, same = _req([*a.prompt, *_ids(12, 70)], 6, 40), _req(a.prompt, 7, 40)
    try:
        sa, sn, ss = run.stream(a), run.stream(nxt), run.stream(same)
        run.go([(_always, sa), (_decoding(sa), sn), (_decoding(sa), ss)])
    finally:
        run.close()
    assert (sa.lane, sn.lane, ss.lane) == (0, 1, 2)
    assert moves == [("copy_rows", 0, 1), ("copy_rows", 0, 2)]
    run.same(oracle, sa, "the stream whose rows were copied")
    run.same(oracle, sn, "a next turn from a busy lane's rows", len(a.prompt) - 1)
    run.same(oracle, ss, "an identical prompt from a busy lane's rows", len(a.prompt) - 1)
    assert sorted(x.lane for x in kept.cache) == [0, 1], "the copied-from entry stays with its lane"
    _accounted(kept)


def test_an_entry_saved_from_one_lane_loads_into_another(tiny, oracle, rig, moves):
    """F4: rows saved from lane 1 (refilled once every free lane holds kept rows) load into lane 2, both through
    their own lane's view."""

    kept = _kept(rig)
    run = Run(_decoder(tiny, rig, kept))
    a, b = _req(_ids(21, 120), 8, 12), _req(_ids(22, 340), 9, 12)
    g0, g1 = _req(_ids(23, 90), None, 15, draft=False), _req(_ids(24, 70), 10, 300, draft=False)
    h, e = _req(_ids(25, 200), 11, 120), _req(_ids(26, 180), None, 120)
    try:
        sa, sb, sg0, sg1, sh, se = (run.stream(r) for r in (a, b, g0, g1, h, e))
        run.go([(_always, sa), (_always, sb)])
        assert (sa.lane, sb.lane) == (0, 1) and moves == []
        f = _req([*b.prompt, *run.got[id(sb)], *_ids(27, 30)], 12, 40)
        sf = run.stream(f)
        run.go([(_always, sg0), (_always, sg1), (_always, sh), (_always, se), (_then(sg0), sf)])
    finally:
        run.close()
    assert (sg0.lane, sg1.lane, sh.lane, se.lane, sf.lane) == (2, 3, 0, 1, 2)
    assert moves == [("save_rows", 0), ("save_rows", 1), ("load_rows", 2)]
    for s, what in ((sa, "a"), (sb, "b"), (sg0, "g0"), (sg1, "g1"), (sh, "h, over a's rows"), (se, "e, over b's")):
        run.same(oracle, s, what)
    run.same(oracle, sf, "b's next turn, loaded into another lane", len(b.prompt) - 1)
    _accounted(kept)


def test_an_entry_in_a_lane_being_refilled_is_evicted_and_the_rest_resume(tiny, oracle, rig, moves):
    """Two entries allowed: a new prompt refilling the lane used least recently evicts the entry living there; the
    other lane's entry resumes in place, and a resend of the evicted prompt runs fresh."""

    kept = _kept(rig, entries=2)
    run = Run(_decoder(tiny, rig, kept))
    a, b = _req(_ids(31, 80), 13, 10), _req(_ids(32, 400), 14, 10)
    x, y = _req(_ids(33, 60), 15, 400, draft=False), _req(_ids(34, 50), None, 400, draft=False)
    c = _req(_ids(35, 150), 16, 30)
    try:
        sa, sb, sx, sy, sc = (run.stream(r) for r in (a, b, x, y, c))
        run.go([(_always, sa), (_always, sb)])
        assert [x.snap.ids for x in kept.cache] == [list(a.prompt[:-1]), list(b.prompt[:-1])]
        nxt, again = _req([*b.prompt, *run.got[id(sb)], *_ids(36, 25)], 17, 20), _req(a.prompt, 18, 20)
        sn, sg = run.stream(nxt), run.stream(again)
        run.go([(_always, sx), (_always, sy), (_decoding(sx, sy), sc), (_then(sc), sn), (_then(sn), sg)])
    finally:
        run.close()
    assert (sx.lane, sy.lane, sc.lane, sn.lane, sg.lane) == (2, 3, 0, 1, 0), "c refills the lane used least recently"
    assert moves == [], "a's entry was evicted, not saved"
    for s, what in ((sa, "a"), (sb, "b"), (sx, "x"), (sy, "y"), (sc, "c, over a's evicted entry"),
                    (sg, "a again, its entry gone")):
        run.same(oracle, s, what)
    run.same(oracle, sn, "b's next turn, its entry kept", len(b.prompt) - 1)
    _accounted(kept)


def test_a_reserve_while_a_same_prompt_resend_fills_spares_its_hit(tiny, oracle, rig, moves):
    """F5/H8: one entry allowed; the resend's hit is pinned, so a new prompt's reserve finds no room instead."""

    kept = _kept(rig, entries=1)
    run = Run(_decoder(tiny, rig, kept))
    a, x = _req(_ids(41, 230), 19, 15), _req(_ids(42, 100), 20, 300, draft=False)
    again, n = _req(a.prompt, 21, 30), _req(_ids(43, 310), 22, 30)
    pins = []
    try:
        sa, sx, sg, sn = (run.stream(r) for r in (a, x, again, n))
        run.go([(_always, sa)])
        hit = kept.cache[0].snap
        run.go([(_always, sx), (_decoding(sx), sg), (_decoding(sx), sn)],
               until=lambda: getattr(sn, "fill", None) is not None and not pins and pins.append(dict(kept.pinned)))
    finally:
        run.close()
    assert pins[0][sg.lane].hit is hit and pins[0][sg.lane].at is None
    assert pins[0][sn.lane].hit is None and pins[0][sn.lane].at is None, "no room for n's snapshot"
    for s, what, cached in ((sa, "a", 0), (sx, "x", 0), (sg, "a's resend", len(a.prompt) - 1), (sn, "n", 0)):
        run.same(oracle, s, what, cached)
    assert len(kept.cache) == 1 and kept.cache[0].snap is hit and moves == []
    _accounted(kept)


def test_drop_settles_every_fill_and_a_later_request_is_exact(tiny, oracle, rig, moves):
    """A drop while a new prompt fills (its span reserved) settles every fill and empties the live ids; the arena
    then holds the entries' spans only, and the same requests again equal their fresh runs."""

    kept = _kept(rig)
    dec = _decoder(tiny, rig, kept)
    run = Run(dec)
    a, x, n = _req(_ids(51, 140), 23, 12), _req(_ids(53, 90), 25, 200, draft=False), _req(_ids(54, 700), 26, 20)
    try:
        sa, sx, sn = (run.stream(r) for r in (a, x, n))
        run.go([(_always, sa)])
        nxt = _req([*a.prompt, *run.got[id(sa)], *_ids(55, 20)], 27, 20)
        sr = run.stream(nxt)
        run.go([(_always, sx), (_decoding(sx), sr), (_decoding(sx), sn)],
               until=lambda: getattr(sn, "fill", None) is not None and sn.fill.i == 1)
        assert sr.lane == 0 and sr.cached == len(a.prompt) - 1 and sn.lane == 2 and kept.pinned[2].at is not None
        filling = any(s is sr for s in dec.filling)
        assert len(kept.pinned) == 1 + filling and len(dec.drop()) == 3
        assert not kept.pinned and kept.live == {}
        _accounted(kept)
        again, later = run.stream(nxt), run.stream(n)
        run.go([(_always, again), (_always, later)])
    finally:
        run.close()
    assert moves == []
    run.same(oracle, sa, "a")
    run.same(oracle, again, "a's next turn after a mid-fill drop: its lane's rows no longer count")
    run.same(oracle, later, "the prompt dropped mid-fill, again")
    _accounted(kept)


def test_three_resends_of_one_conversation_across_lanes(tiny, oracle, rig, graphs, moves):
    kept = _kept(rig, entries=entries_wanted({}, lanes=S))
    run = Run(_decoder(tiny, rig, kept, graphs))
    x, t = _req(_ids(61, 110), None, 600, draft=False), _req(_ids(62, 280), 29, 16)
    try:
        sx, st = run.stream(x), run.stream(t)
        run.go([(_always, sx), (_always, st)], until=_then(st))
        reply = run.got[id(st)]
        r1, r2, r3 = (_req(t.prompt, 30, 30), _req([*t.prompt, *reply, *_ids(63, 33)], 31, 30),
                      _req([*t.prompt, *reply, *_ids(64, 47)], None, 30))
        s1, s2, s3 = (run.stream(r) for r in (r1, r2, r3))
        run.go([(_always, s1), (_always, s2), (_always, s3)], until=_then(s1, s2, s3))
        r4 = _req([*r2.prompt, *run.got[id(s2)], *_ids(65, 21)], 32, 30)
        s4 = run.stream(r4)
        run.go([(_always, s4)])
    finally:
        run.close()
    assert (sx.lane, st.lane, s1.lane, s2.lane, s3.lane, s4.lane) == (0, 1, 1, 2, 3, 2)
    assert moves == [("copy_rows", 1, 2), ("copy_rows", 1, 3)]
    run.same(oracle, sx, "the decoding lane")
    run.same(oracle, st, "the first turn")
    for s, what in ((s1, "the identical resend, in place"), (s2, "a next turn, copied"),
                    (s3, "another next turn, copied")):
        run.same(oracle, s, what, len(t.prompt) - 1)
    run.same(oracle, s4, "the third turn, in place over the second's rows", len(r2.prompt) - 1)
    _accounted(kept)
