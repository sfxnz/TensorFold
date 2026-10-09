"""Greedy draws on the device: each row's largest value at its lowest id over every rank's columns (-0 tying +0), on
one rank and two thread ranks, for every row count a verify forward holds, with values tied inside a shard, across the
shards, at zero and at infinities; a row draws alike alone and beside others; a graph's pinned read equals the eager
keys; ``target_rows`` draws the same wherever each shard's largest value is unique; and with GREEDY_ENV on, solo replies
(graphs, eager, two ranks) and lane replies (graphs, eager) equal the replies ``target_rows`` draws.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
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
from tensorfold.families.deepseek_v41.cuda import BLOCK, MAX_LANES, MAX_ROWS, PREFILL_ROWS, dspark, loader, sample
from tensorfold.families.deepseek_v41.cuda import decode as D
from tensorfold.families.deepseek_v41.cuda import multi as M
from tensorfold.families.deepseek_v41.cuda import prefill as P
from tensorfold.families.deepseek_v41.cuda.buffers import Buffers
from tensorfold.families.deepseek_v41.cuda.graphs import LaneGraphs
from tensorfold.families.deepseek_v41.cuda.lanes import Ahead, Lanes
from tensorfold.families.deepseek_v41.engram_table import Reader

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
VOCAB = 6000
SPLIT = 2600                            # rank 0's shard: an uneven split
ROWS = range(1, MAX_ROWS * MAX_LANES + 1)       # every verify forward's row count
KINDS = ("spread", "tied in a shard", "tied across shards", "zeros", "infinities")
ONE = SimpleNamespace(comm=None, world=1, vocab_offset=0)
CAP = 1024
S = 4                                   # lanes
REPLY = 40
PROMPTS = (5, 100, 600, 300)            # the 16/32-entry transitions, the ring's wrap, full lists
POLICIES = [None, (1, None), (3, None), (BLOCK, None), (BLOCK, 0.3)]
KEYED = Sampling(seed=7, temperature=1.0, top_k=20, top_p=0.95)
NUCLEUS = Sampling(seed=8, temperature=1.0, top_k=0, top_p=0.95)


def _rows(seed: int, n: int, kind: str) -> torch.Tensor:
    """fp32 logits [n, VOCAB] of ``kind`` on the device."""

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, VOCAB, generator=g) * 3.0
    top = float(x.max()) + 1.0
    for r in range(n):
        if kind == "tied in a shard":           # the best at several ids of each shard
            x[r, [7 + r, 30, SPLIT - 1, SPLIT + 5 + r, VOCAB - 1]] = top
        elif kind == "tied across shards":      # one best a shard, equal across the two
            x[r, [SPLIT - 1 - r, SPLIT + r]] = top
        elif kind == "zeros":                   # the best is zero: -0 at the lowest id on odd rows, +0 on even ones
            x[r] = -x[r].abs() - 1.0
            x[r, [40 + r, SPLIT + 3]] = 0.0
            x[r, [11 + r if r % 2 else 60 + r, SPLIT - 2]] = -0.0
        elif kind == "infinities":              # +inf twice on odd rows, -inf on even ones, row 0 all -inf
            x[r, [SPLIT + 9 + r, 500]] = float("inf") if r % 2 else -float("inf")
    if kind == "infinities":
        x[0] = -float("inf")
    return x.cuda()


def _reference(x: torch.Tensor) -> list[int]:
    """Each row's lowest id holding its largest value (numpy's == ties -0 with +0)."""

    return [int(np.flatnonzero(r == r.max())[0]) for r in x.cpu().numpy()]


def _drawn(w, x: torch.Tensor) -> list[int]:
    R = x.shape[0]
    keys = torch.empty((3, w.world * R), dtype=torch.int64, device=x.device)
    return [k & 0xFFFFFFFF for k in sample.least_keys(w, x, keys[0, :R], keys[1], keys[2, :R]).tolist()]


def _today(w, x: torch.Tensor) -> list[int]:
    return sample.target_rows(w, x, list(range(x.shape[0])), None)[0]


def _two(x: torch.Tensor, fn) -> list[int]:
    """``fn(w, shard)`` on two thread ranks, each holding its shard; both ranks agree."""

    def rank(r: int):
        def go(comm):
            w = SimpleNamespace(comm=comm, world=2, vocab_offset=0 if r == 0 else SPLIT)
            return fn(w, (x[:, :SPLIT] if r == 0 else x[:, SPLIT:]).contiguous())
        return go

    a, b = run_pair(rank(0), rank(1))
    assert a == b, "the two ranks drew differently"
    return a


@pytest.mark.parametrize("kind", KINDS)
def test_each_row_draws_its_largest_value_at_its_lowest_id(kind):
    for n in ROWS:
        x = _rows(n, n, kind)
        want = _reference(x)
        assert _drawn(ONE, x) == want, f"one rank, {n} rows"
        assert _two(x, _drawn) == want, f"two ranks, {n} rows"


@pytest.mark.parametrize("kind", KINDS)
def test_a_row_draws_alike_alone_and_beside_others(kind):
    x = _rows(99, len(ROWS), kind)
    full = _drawn(ONE, x)
    for n in ROWS:
        assert _drawn(ONE, x[:n]) == full[:n], f"the first {n} rows"
        assert _drawn(ONE, x[n - 1:n]) == full[n - 1:n], f"row {n - 1} alone"


@pytest.mark.parametrize("kind", ["spread", "tied across shards"])
def test_target_rows_draws_the_same_where_each_shard_has_one_best(kind):
    for n in ROWS:
        x = _rows(n, n, kind)
        if kind == "spread":                    # one rank's shard is the whole row
            assert _drawn(ONE, x) == _today(ONE, x), f"one rank, {n} rows"
        assert _two(x, _drawn) == _two(x, _today), f"two ranks, {n} rows"


def test_a_graph_reads_the_eager_keys_back_pinned():
    n = len(ROWS)
    b = SimpleNamespace(logits=torch.zeros((n, VOCAB), device="cuda"), gkeys=torch.zeros((n,), dtype=torch.int64,
                        device="cuda"), ggot=torch.zeros((n,), dtype=torch.int64, device="cuda"),
                        gbest=torch.zeros((n,), dtype=torch.int64, device="cuda"),
                        gkeys_host=torch.zeros((n,), dtype=torch.int64, pin_memory=True))
    graphs = {}
    for R in (1, MAX_ROWS, n):
        sample.window_keys(ONE, b, R)           # compiled outside the capture
        torch.cuda.synchronize()
        graphs[R] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[R]):
            sample.window_keys(ONE, b, R)
    for kind in KINDS:
        x = _rows(5, n, kind)
        b.logits.copy_(x)
        want = _reference(x)
        for R, g in graphs.items():
            b.gkeys_host.fill_(-1)
            g.replay()
            torch.cuda.current_stream().synchronize()
            assert sample.window_rows(b, 0, R) == want[:R], f"{kind}, {R} rows"
            assert sample.window_rows(b, 1, R - 1) == want[1:R], f"{kind}, rows 1..{R}"


def _ids(seed: int, n: int, vocab: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, vocab, (n,), generator=g).tolist()


def _reply(e: D.Engine, prompt, sampling, policy) -> list[int]:
    first = P.prefill(e, prompt, sampling)
    if policy is None:
        return D.serial_decode(e, first, REPLY, sampling, stop_eos=False).tokens
    return D.dspark_decode(e, first, REPLY, sampling, drafts=policy[0], confidence=policy[1], stop_eos=False).tokens


@pytest.fixture(scope="module")
def ref(tiny_dir):
    return RefWeights(tiny_dir)


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    return loader.load(tiny_dir, Config.read(tiny_dir), 0, 1, None, capacity=CAP)


@pytest.fixture(scope="module")
def engines(tiny, ref):
    """(``target_rows``' graphed engine, the device draw's graphed and eager engines)."""

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(sample.GREEDY_ENV, "0")
        plain = D.Engine(tiny, CAP, graphs=True, hasher=ref.hasher, reader=ref.reader)
        mp.setenv(sample.GREEDY_ENV, "1")
        on = [D.Engine(tiny, CAP, graphs=g, hasher=ref.hasher, reader=ref.reader) for g in (True, False)]
    assert plain.dbuf.gkeys is None and all(e.dbuf.gkeys is not None for e in on)
    return plain, on


@pytest.mark.parametrize("seed", range(len(PROMPTS)))
def test_solo_replies_equal_the_target_rows_replies(engines, seed):
    plain, on = engines
    prompt = _ids(seed, PROMPTS[seed], plain.w.cfg.vocab_size)
    for sampling in (None, KEYED, NUCLEUS):
        want = _reply(plain, prompt, sampling, None)
        for e, how in zip(on, ("graphs", "eager")):
            for policy in POLICIES:
                assert _reply(e, prompt, sampling, policy) == want, f"{how}, {policy}, {sampling}"
    assert on[1].replays["graph"] == 0


@pytest.mark.parametrize("seed", (0, 2))
def test_two_ranks_draw_the_target_rows_replies(tiny_dir, tiny, ref, seed):
    comms = pair()
    ws = [loader.load(tiny_dir, tiny.cfg, r, 2, comms[r], capacity=CAP) for r in range(2)]
    engs = {}
    with pytest.MonkeyPatch.context() as mp:
        for on in (False, True):
            mp.setenv(sample.GREEDY_ENV, "1" if on else "0")
            engs[on] = [D.Engine(ws[r], CAP, hasher=ref.hasher, reader=Reader(ref.reader.layout)) for r in range(2)]
    prompt = _ids(seed, PROMPTS[seed], tiny.cfg.vocab_size)

    def rank(r):
        def go(_):
            return [[_reply(engs[on][r], prompt, None, policy) for policy in POLICIES] for on in (False, True)]
        return go

    got0, got1 = run_pair(rank(0), rank(1), comms)
    assert got0 == got1, "the ranks decoded differently"
    off, on = got0
    assert all(x == off[0] for x in off) and on == off


def _lane_decoder(w, ref, policy, graphed: bool) -> M.LaneDecoder:
    """A lane decoder over S lanes whose shared forward draws greedy rows on the device."""

    cfg, dev = w.cfg, w.device
    lanes = Lanes(cfg, S, CAP, dev)
    pbuf = Buffers(cfg, w.world, min(PREFILL_ROWS, CAP), CAP, prefill=True, device=dev)
    mbuf = Buffers(cfg, w.world, MAX_ROWS * S, CAP, device=dev, lanes=S, greedy=True)
    engines = [D.Engine(w, CAP, hasher=ref.hasher, reader=ref.reader, st=lanes.view(k), pbuf=pbuf, dbuf=mbuf,
                        dwork=dspark.Work(cfg, w.world, dev), ahead=Ahead(cfg, w.world)) for k in range(S)]
    graphs = LaneGraphs(w, lanes, mbuf, engines) if graphed else None
    if graphs is not None:
        graphs.warm()
    return M.LaneDecoder(w, lanes, engines, mbuf, pbuf, policy, (cfg.eos_token_id,), graphs, 0.5)


@pytest.mark.parametrize("graphed", [True, False], ids=["graphs", "eager"])
@pytest.mark.parametrize("policy", [(3, None), (BLOCK, 0.3)])
def test_lanes_together_equal_the_target_rows_replies(tiny, ref, engines, policy, graphed):
    """Greedy lanes beside a top_k lane and a top_k-off lane, one serial (``"draft": false``)."""

    plain = engines[0]
    samplings, drafts = (None, None, KEYED, NUCLEUS), (True, False, True, True)
    prompts = [_ids(10 + k, n, tiny.cfg.vocab_size) for k, n in enumerate(PROMPTS)]
    want = [_reply(plain, p, s, None) for p, s in zip(prompts, samplings)]
    dec = _lane_decoder(tiny, ref, policy, graphed)
    got: list[list[int]] = [[] for _ in prompts]
    for p, s, d, out in zip(prompts, samplings, drafts, got):
        stream = Stream(p, REPLY, s, draft=d, stop_eos=False)
        stream.emit = out.extend
        dec.admit(stream)
    steps = 0
    while dec.live():
        dec.finish(dec.round())
        steps += 1
        assert steps < 5000, "the decoder made no progress"
    assert got == want
