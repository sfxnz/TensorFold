"""Concurrent requests over 4 lanes on two thread ranks: rank 0 drives the lane decoder as the Scheduler does and tells
rank 1 every admission, round, finish and drop over the link (its store fallback, in memory). Streams together equal
the same streams one at a time and their solo runs on the legacy two-rank engine (tokens, each verify window's logits
bits on rank 0, drafted, accepted and rounds) at 1, 2 and 4 streams, greedy and keyed; a stream joins mid-flight;
clients leave while a prompt fills, at the first token and mid-decode; a resumed prompt equals its fresh run while
others decode; a tampered plan and a missing snapshot are refused on both ranks; failed rounds drop on both ranks; and
each time rank 1 decoded what rank 0 did and ends with every lane free.
"""

from __future__ import annotations

import hashlib
import threading
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_tiny
from dsv41_pair import pair, run_pair
from dsv41_ref_weights import RefWeights

from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import BLOCK, MAX_ROWS, dspark, loader, snapshot
from tensorfold.families.deepseek_v41.cuda import decode as D
from tensorfold.families.deepseek_v41.cuda import multi as M
from tensorfold.families.deepseek_v41.cuda import multi_tp as T
from tensorfold.families.deepseek_v41.cuda import prefill as P
from tensorfold.families.deepseek_v41.cuda.buffers import Buffers
from tensorfold.families.deepseek_v41.cuda.cache import Kept
from tensorfold.families.deepseek_v41.cuda.lanes import Ahead, Lanes
from tensorfold.families.deepseek_v41.engram_table import Reader

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
CAP = 1024
S = 4                                   # lanes
ROWS = 129                              # prompt span rows: longer prompts fill in several spans
POLICY = (BLOCK, 0.3)                   # the confidence policy
LENGTHS = (5, 300, 100, 600)            # the 16/32-entry transitions, the ring's wrap, full lists
COUNTS = (24, 17, 30, 20)               # lanes finish in different rounds


class Req(NamedTuple):
    prompt: tuple[int, ...]
    sampling: Sampling | None
    count: int
    draft: bool = True


def _ids(seed: int, n: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, 1024, (n,), generator=g).tolist()


def _req(seed: int, n: int, keyed: bool, count: int, draft: bool = True, prompt=None) -> Req:
    sampling = Sampling(seed=seed, temperature=1.0, top_k=20 * (seed % 2), top_p=0.95) if keyed else None
    return Req(tuple(_ids(seed, n) if prompt is None else prompt), sampling, count, draft)


def _bits(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class _Store:
    """The TCP store's set / wait / get / delete_key, in memory."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.cv = threading.Condition()

    def set(self, key, value) -> None:
        with self.cv:
            self.data[key] = value.encode() if isinstance(value, str) else value
            self.cv.notify_all()

    def wait(self, keys, timeout=None) -> None:
        with self.cv:
            if not self.cv.wait_for(lambda: all(k in self.data for k in keys), timeout=60):
                raise TimeoutError(keys)

    def get(self, key) -> bytes:
        return self.data[key]

    def delete_key(self, key) -> None:
        with self.cv:
            self.data.pop(key, None)


@pytest.fixture(scope="module")
def ref(tiny_dir):
    return RefWeights(tiny_dir)


@pytest.fixture(scope="module")
def ranks(tiny_dir, ref):
    """Per rank: (weights, its legacy oracle Engine, its lanes, shared buffers and lane Engines); and the comms."""

    cfg, comms = Config.read(tiny_dir), pair()
    out = []
    for r in range(2):
        w = loader.load(tiny_dir, cfg, r, 2, comms[r], capacity=CAP)
        reader, dev = Reader(ref.reader.layout), w.device
        lanes = Lanes(cfg, S, CAP, dev)
        pbuf = Buffers(cfg, w.world, ROWS, CAP, prefill=True, device=dev)
        mbuf = Buffers(cfg, w.world, MAX_ROWS * S, CAP, device=dev, lanes=S)
        engines = [D.Engine(w, CAP, hasher=ref.hasher, reader=reader, st=lanes.view(k), pbuf=pbuf, dbuf=mbuf,
                            dwork=dspark.Work(cfg, w.world, dev), ahead=Ahead(cfg, w.world)) for k in range(S)]
        out.append((w, D.Engine(w, CAP, hasher=ref.hasher, reader=reader), lanes, mbuf, pbuf, engines))
    return out, comms


def _fresh(comms) -> tuple:
    """The pair's barrier, usable again after an earlier failure broke it."""

    if comms[0].hub.gate.broken:
        comms[0].hub.gate.reset()
    return comms


_SOLO: dict = {}


def _solo(ranks, req: Req) -> dict:
    """The legacy engine's fresh run of ``req`` on both ranks: tokens, rank 0's verify windows, draft stats."""

    if req in _SOLO:
        return _SOLO[req]
    rigs, comms = ranks

    def rank(r):
        def go(_):
            e, windows, forward = rigs[r][1], [], rigs[r][1].forward

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
            return {"tokens": res.tokens, "windows": windows, "stats": (res.drafted, res.accepted, res.rounds)}
        return go

    a, b = run_pair(rank(0), rank(1), _fresh(comms))
    assert a["tokens"] == b["tokens"] and a["stats"] == b["stats"], "the legacy ranks decoded differently"
    _SOLO[req] = a
    return a


class Run:
    """Rank 0's decoder driven as the Scheduler drives it: each stream admitted once its condition holds and a lane is
    free (a refusal is its error), then a round (a failure drops every live stream) and ``finish``, a step."""

    def __init__(self, dec: M.LaneDecoder) -> None:
        self.dec, self.got, self.windows, self.reqs, self.all = dec, {}, {}, {}, []
        dec._draws = self._recorder(dec._draws)

    def _recorder(self, draws):
        def rec(live):
            for s in live:
                self.windows.setdefault(id(s), []).append([_bits(row) for row in self.dec.mbuf.logits[s.seg0:s.seg0 + s.R]])
            return draws(live)
        return rec

    def stream(self, req: Req, stop=None) -> Stream:
        """``req`` as a Stream; ``stop(tokens so far)`` True makes its client leave."""

        s = Stream(list(req.prompt), req.count, req.sampling, draft=req.draft, stop_eos=False)
        got = self.got[id(s)] = []
        s.emit = lambda new: (got.extend(new), stop is not None and stop(got))[1]
        self.reqs[id(s)] = req
        self.all.append(s)
        return s

    def go(self, arrivals: list, each=None) -> None:
        """Admit each (condition, stream) once its condition holds, ``each(self)`` before every round, until nothing
        is live."""

        waiting, steps = list(arrivals), 0
        while waiting or self.dec.live():
            for cond, s in list(waiting):
                if self.dec.live() < S and cond():
                    waiting.remove((cond, s))
                    try:
                        self.dec.admit(s)
                    except Exception as exc:        # noqa: BLE001  (the Scheduler replies with the error)
                        s.error, s.done = exc, True
            if each is not None:
                each(self)
            try:
                done = self.dec.round()
            except Exception as exc:                # noqa: BLE001  (the Scheduler drops the live streams)
                done = []
                for s in self.dec.drop():
                    s.error, s.done = exc, True
            self.dec.finish(done)
            steps += 1
            assert steps < 5000, "the decoder made no progress"

    def same(self, ranks, s: Stream, what: str, cached: int = 0, cut: bool = False) -> None:
        """``s`` equals its fresh solo run (a stream cut short: its prefix), having resumed ``cached`` tokens."""

        want = _solo(ranks, self.reqs[id(s)])
        got, windows = self.got[id(s)], self.windows.get(id(s), [])
        assert s.done and s.error is None, f"{what}: {s.error}"
        assert s.cached == cached, f"{what}: resumed from {s.cached} tokens, not {cached}"
        if cut:
            assert got == want["tokens"][:len(got)] and windows == want["windows"][:len(windows)], what
            return
        assert got == want["tokens"], f"{what}: tokens"
        assert windows == want["windows"], f"{what}: a verify window's logits bits"
        assert (s.drafted, s.accepted, s.rounds) == want["stats"], f"{what}: drafted, accepted, rounds"

    def close(self) -> None:
        del self.dec._draws


def _kept(engines, entries: int) -> Kept:
    e = engines[0]
    return Kept(engines, 16 * (snapshot.state_bytes(e) + snapshot._span(e.st.row_views(CAP))[1]), entries)


def _two(ranks, drive, *, entries: int = 16, edit=None, setup=None) -> tuple[Run, list]:
    """``drive(run)`` on rank 0 while rank 1 follows (``edit(op)`` alters each op it receives) -> (rank 0's Run,
    rank 1's refusals); rank 1 admitted and decoded what rank 0 did, and ends with every lane free."""

    rigs, comms = ranks
    decs = [M.LaneDecoder(w, lanes, engines, mbuf, pbuf, POLICY, (w.cfg.eos_token_id,), None, 0.5,
                          kept=_kept(engines, entries))
            for w, _, lanes, mbuf, pbuf, engines in rigs]
    if setup is not None:
        setup(decs)
    admitted, refused, admit = [], [], decs[1].admit

    def admit1(s, told=None):
        try:
            admit(s, told=told)
        except Exception as exc:
            refused.append(exc)
            raise
        admitted.append(s)

    decs[1].admit = admit1
    store, run = _Store(), Run(decs[0])

    def lead(_):
        decs[0].link = T.Link(store)
        try:
            drive(run)
        finally:
            decs[0].link.send(["stop"])

    def follow(_):
        link = T.Link(store)
        if edit is not None:
            receive = link.receive
            link.receive = lambda: edit(receive())
        decs[1].follow(link)

    try:
        run_pair(lead, follow, _fresh(comms))
    finally:
        run.close()
    one = decs[1]
    assert not one.streams and not one.filling and one._free() == list(range(S)) and not one.kept.pinned
    assert one.follower is None and one.next_id == decs[0].next_id
    by_sid = {s.sid: s for s in admitted}
    for s0 in run.all:
        if hasattr(s0, "lane"):                     # admitted on rank 0
            s1 = by_sid[s0.sid]
            assert (s1.out, s1.cached, s1.lane) == (s0.out, s0.cached, s0.lane), f"rank 1's stream {s0.sid}"
    return run, refused


def _always() -> bool:
    return True


def _then(*streams: Stream):
    """A condition: every one of ``streams`` has ended."""

    return lambda: all(s.done for s in streams)


def _decoding(*streams: Stream):
    """A condition: every one of ``streams`` decodes."""

    return lambda: all(s.out and not s.done for s in streams)


@pytest.mark.parametrize("keyed", [False, True], ids=["greedy", "keyed"])
@pytest.mark.parametrize("c", [1, 2, 4])
def test_streams_together_equal_one_at_a_time_and_solo(ranks, c, keyed):
    """c streams admitted at once, and the same streams one after another, each equal to its solo run; at 4 one
    lane is serial (``"draft": false``)."""

    reqs = [_req(10 * c + j + 100 * keyed, LENGTHS[j], keyed, COUNTS[j], draft=j != 3) for j in range(c)]

    def together(run):
        run.go([(_always, run.stream(r)) for r in reqs])

    def serial(run):
        for r in reqs:
            run.go([(_always, run.stream(r))])

    for drive, how in ((together, "together"), (serial, "one at a time")):
        run, refused = _two(ranks, drive)
        assert len(run.all) == c and not refused
        for j, s in enumerate(run.all):
            run.same(ranks, s, f"stream {j} of {c}, {how}")


def test_a_stream_joins_mid_flight(ranks):
    """Two streams decode; two more arrive, one filling in spans beside the rounds, one waiting for its lane."""

    a, b = _req(21, 40, True, 40), _req(22, 120, False, 40, draft=False)
    x, y = _req(23, 450, True, 20), _req(24, 260, True, 20)

    def drive(run):
        sa, sb, sx, sy = (run.stream(r) for r in (a, b, x, y))
        run.go([(_always, sa), (_always, sb), (_decoding(sa, sb), sx), (lambda: getattr(sx, "fill", None) is not None, sy)])

    run, _ = _two(ranks, drive)
    for s, what in zip(run.all, ("a", "b", "x, joining mid-decode", "y, arriving while x fills")):
        run.same(ranks, s, what)


def test_clients_that_leave_free_their_lanes_on_both_ranks(ranks):
    """The server ends one stream while its prompt fills, a client leaves at its first token, another mid-decode;
    the stream that stays and a later request in a freed lane are exact."""

    reqs = [_req(31, 600, True, 40), _req(32, 90, True, 40), _req(33, 200, False, 40), _req(34, 60, True, 30)]
    later = _req(35, 300, True, 25)
    stops = [None, lambda got: True, lambda got: len(got) >= 12, None]
    ended = []

    def drive(run):
        streams = [run.stream(r, stop) for r, stop in zip(reqs, stops)]
        last = run.stream(later)

        def each(run):
            s = streams[0]
            if not ended and s.fill is not None and s.fill.i >= 1:
                ended.append(s.fill.i)
                run.dec.finish([s])                 # the server ends it mid-fill
                s.done = True

        run.go([*((_always, s) for s in streams), (_then(*streams[:3]), last)], each)

    run, _ = _two(ranks, drive)
    s0, s1, s2, s3, last = run.all
    assert ended and not run.got[id(s0)] and s0.rounds == 0, "the first ended while its prompt filled"
    assert len(run.got[id(s1)]) == 1 and s1.rounds == 0
    assert 12 <= len(run.got[id(s2)]) < 12 + MAX_ROWS
    run.same(ranks, s1, "a client gone at its first token", cut=True)
    run.same(ranks, s2, "a client gone mid-decode", cut=True)
    run.same(ranks, s3, "the stream whose client stayed")
    run.same(ranks, last, "a later request in a freed lane")


def test_a_resumed_prompt_equals_its_fresh_run_while_others_decode(ranks):
    a, x = _req(41, 260, True, 20), _req(42, 150, True, 300, draft=False)
    later = {}

    def drive(run):
        sa, sx = run.stream(a), run.stream(x)
        run.go([(_always, sa)])
        later["next"] = nxt = _req(43, 40, True, 30, prompt=[*a.prompt, *run.got[id(sa)], *_ids(44, 40)])
        later["again"] = again = _req(45, 0, True, 25, prompt=a.prompt)
        sn, sg = run.stream(nxt), run.stream(again)
        run.go([(_always, sx), (_decoding(sx), sn), (_decoding(sx), sg)])

    run, _ = _two(ranks, drive)
    sa, sx, sn, sg = run.all
    run.same(ranks, sa, "the first turn")
    run.same(ranks, sx, "the lane decoding beside the resumes")
    run.same(ranks, sn, "the next turn, resumed", len(a.prompt) - 1)
    run.same(ranks, sg, "an identical resend, resumed", len(a.prompt) - 1)


def test_a_tampered_plan_and_a_missing_snapshot_are_refused_on_both_ranks(ranks):
    """Rank 1 told another span size, then rank 1 missing the snapshot: both ranks refuse each before any collective,
    and the same request next resumes exactly, while another stream decodes throughout."""

    a, x = _req(51, 230, True, 15), _req(52, 100, False, 400, draft=False)
    resend, admits = _req(53, 0, True, 20, prompt=a.prompt), []

    def edit(op):
        if op is not None and op[0] == "admit":
            admits.append(op)
            if len(admits) == 3:                    # the first resend: a smaller span on rank 1
                op[-1]["rows"] = 64
        return op

    def setup(decs):
        find = decs[1].kept.find

        def missing(prompt, cached):
            if len(admits) == 5:                    # the second resend
                raise RuntimeError("rank 1 lost it")
            return find(prompt, cached)

        decs[1].kept.find = missing

    def drive(run):
        sa, sx = run.stream(a), run.stream(x)
        run.go([(_always, sa)])
        streams = [run.stream(resend) for _ in range(4)]
        run.go([(_always, sx), *((_decoding(sx), s) for s in streams)])

    run, refused = _two(ranks, drive, edit=edit, setup=setup)
    sa, sx, tampered, ok1, missing, ok2 = run.all
    assert isinstance(tampered.error, T.OutOfStep) and "planned different admissions" in str(tampered.error)
    assert isinstance(missing.error, T.OutOfStep) and "no snapshot of the 229 tokens" in str(missing.error)
    assert [str(e) for e in refused] == [str(tampered.error), str(missing.error)], "rank 1 refused both too"
    run.same(ranks, sa, "a")
    run.same(ranks, sx, "the stream decoding throughout")
    for s, what in ((ok1, "the resend after a tampered plan"), (ok2, "the resend after a missing snapshot")):
        run.same(ranks, s, what, len(a.prompt) - 1)


@pytest.mark.parametrize("how", ["symmetric", "agreement"])
def test_a_failed_round_drops_on_both_ranks_and_serving_goes_on(ranks, how, capfd):
    """The third decode round fails on both ranks before its collectives, or its plan reaches rank 1 altered and the
    agreement refuses it: both ranks drop every live stream, and the same requests then run exactly."""

    reqs = [_req(61, 80, True, 30), _req(62, 200, False, 30), _req(63, 40, True, 30, draft=False)]
    rounds = []

    def edit(op):
        if op is not None and op[0] == "round" and op[2]["decode"]:
            rounds.append(op)
            if how == "agreement" and len(rounds) == 3:
                op[2]["spans"] = 2
        return op

    def setup(decs):
        if how != "symmetric":
            return
        for dec in decs:
            calls, real = [], dec._decode

            def fails(calls=calls, real=real):
                calls.append(1)
                if len(calls) == 3:
                    raise RuntimeError("an injected round failure")
                return real()

            dec._decode = fails

    def drive(run):
        first = [run.stream(r) for r in reqs]
        run.go([(_always, s) for s in first])
        run.go([(_always, run.stream(r)) for r in reqs])

    run, _ = _two(ranks, drive, edit=edit, setup=setup)
    first, again = run.all[:3], run.all[3:]
    want = T.OutOfStep if how == "agreement" else RuntimeError
    assert all(isinstance(s.error, want) for s in first), [s.error for s in first]
    assert "rank 1: a round failed" in capfd.readouterr().out
    for j, s in enumerate(again):
        run.same(ranks, s, f"request {j} after the drop")
