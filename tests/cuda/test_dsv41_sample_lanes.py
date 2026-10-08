"""A round's target draws over its lanes: each lane's tokens equal its own ``target_rows`` call, on one rank and on two
thread ranks, top_k-off lanes drawn together with their own knobs beside greedy and top_k lanes, including values tied
inside a rank's shard and rows that leave the device draw (a cut past TMAX, rows it cannot decide)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from dsv41_pair import run_pair

from tensorfold.cuda import sampling as cs
from tensorfold.cuda.sampling import TMAX
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.cuda import sample

VOCAB = 6000
SPLIT = 2600            # rank 0's shard: an uneven split


def _rows(seed: int, n: int, *, scale: float = 3.0, levels: int = 0) -> torch.Tensor:
    """fp32 logits [n, VOCAB] at bf16 values; ``levels`` > 0 rounds them to that many steps, so many values tie."""

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, VOCAB, generator=g) * scale
    if levels:
        x = torch.round(x * levels / scale) * scale / levels
    return x.to(torch.bfloat16).float().cuda()


def _tied(seed: int, n: int) -> torch.Tensor:
    """Rows whose best value repeats at several ids inside each rank's shard, the rest tied in a few levels."""

    x = _rows(seed, n, levels=4)
    top = float(x.max()) + 1.0
    for r in range(n):
        x[r, [5 + r, 17, SPLIT - 3, SPLIT + 11, SPLIT + 400 + r, VOCAB - 1]] = top
    return x


LANES = [
    (None, 3, "greedy"),
    (Sampling(11, 0.7, 20, 0.9), 2, "top_k 20"),
    (Sampling(12, 0.7, 0, 1.0), 4, "top_k off"),
    (Sampling(13, 1.3, 0, 0.95, 0.05), 6, "top_k off, a cut and min_p"),
    (Sampling(14, 0.33, 0, 0.5), 1, "top_k off, a tight cut"),
    (Sampling(15, 1.0, 0, 1.0, 0.2), 5, "top_k off, min_p"),
]


def _lanes(seed: int, picks, tied: bool = False):
    rows = [(_tied if tied else _rows)(seed + k, n) for k, (_, n, _) in enumerate(picks)]
    positions = [[1000 * k + 37 + r for r in range(n)] for k, (_, n, _) in enumerate(picks)]
    return rows, positions, [s for s, _, _ in picks]


def _one(rows, positions, samplings):
    """(lane_rows, each lane's own target_rows) on one rank."""

    w = SimpleNamespace(comm=None, world=1, vocab_offset=0)
    got, _ = sample.lane_rows(w, rows, positions, samplings)
    return got, [sample.target_rows(w, x, p, s)[0] for x, p, s in zip(rows, positions, samplings)]


def _two(rows, positions, samplings):
    """(lane_rows, each lane's own target_rows) on two thread ranks, each holding its shard; both ranks agree."""

    def rank(r: int):
        def go(comm):
            w = SimpleNamespace(comm=comm, world=2, vocab_offset=0 if r == 0 else SPLIT)
            mine = [(x[:, :SPLIT] if r == 0 else x[:, SPLIT:]).contiguous() for x in rows]
            got, _ = sample.lane_rows(w, mine, positions, samplings)
            return got, [sample.target_rows(w, x, p, s)[0] for x, p, s in zip(mine, positions, samplings)]
        return go

    a, b = run_pair(rank(0), rank(1))
    assert a == b, "the two ranks drew differently"
    return a


@pytest.mark.parametrize("ranks", [_one, _two], ids=["one rank", "two ranks"])
@pytest.mark.parametrize("tied", [False, True], ids=["spread", "tied in a shard"])
def test_each_lane_draws_what_its_own_call_draws(ranks, tied):
    for seed in (0, 100, 200):
        rows, positions, samplings = _lanes(seed, LANES, tied)
        got, own = ranks(rows, positions, samplings)
        assert got == own, f"seed {seed}"
        alone, _ = ranks(rows[2:3], positions[2:3], samplings[2:3])
        assert alone == own[2:3], "a lone top_k-off lane"


@pytest.mark.parametrize("cuts", ["mixed", "cut", "uncut"])
def test_top_k_off_lanes_drawn_together_decide_their_rows_on_the_device(cuts):
    """The batch decides ordinary rows itself, each equal to its lane's own ``nucleus_rows``, whether its lanes all
    cut by top_p, none do, or both kinds share it."""

    picks = [x for x in LANES if x[0] is not None and not x[0].top_k]
    if cuts != "mixed":
        picks = [x for x in picks if (x[0].top_p < 1.0) == (cuts == "cut")]
    assert len(picks) > 1
    rows, positions, samplings = _lanes(7, picks)
    got = sample.nucleus_lanes(rows, positions, samplings)
    assert None not in got
    assert got == [cs.nucleus_rows(x, p, s) for x, p, s in zip(rows, positions, samplings)]


@pytest.mark.parametrize("ranks", [_one, _two], ids=["one rank", "two ranks"])
def test_lanes_the_device_draw_cannot_decide_take_their_own_calls(ranks, monkeypatch):
    """A cut past TMAX (the host rule) never enters the batch; a lane with a row the batch cannot decide is drawn by
    its own call."""

    picks = [(Sampling(21, 2.0 * TMAX, 0, 0.9), 2, "past TMAX"), (Sampling(22, 1.0, 0, 0.95), 3, "undecided"),
             (Sampling(23, 0.8, 0, 1.0), 4, "ordinary")]
    rows, positions, samplings = _lanes(31, picks)
    undecided = positions[1][1]
    batched, decide, nucleus = [], sample._decide, sample.nucleus_lanes
    monkeypatch.setattr(sample, "_decide", lambda *a: None if a[-1] == undecided else decide(*a))
    monkeypatch.setattr(sample, "nucleus_lanes", lambda r, p, s, **k: batched.append(len(r)) or nucleus(r, p, s, **k))
    got, own = ranks(rows, positions, samplings)
    assert got == own
    assert set(batched) == {2}, "the lane past TMAX stays out of the batch"
    assert nucleus(rows[1:], positions[1:], samplings[1:])[0] is None, "the undecided lane is left to its own call"
