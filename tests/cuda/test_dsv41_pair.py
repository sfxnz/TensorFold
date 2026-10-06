"""Two thread ranks on one GPU gather like NCCL, sampling over two vocabulary halves equals one full-vocabulary rank,
and device drafts follow the host rule on either."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from dsv41_pair import run_pair

from tensorfold.cuda import comm as C
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.cuda import sample as S

VOCAB = 129280                     # the checkpoint's vocabulary: 64,640 head rows per rank
HALF = VOCAB // 2
POSITIONS = [517, 518, 519, 520, 521, 522]
SAMPLINGS = [None] + [Sampling(seed=seed, temperature=t, top_k=k, top_p=p)
                      for seed in (0, 7, 2 ** 61 + 3) for t in (0.0, 1.0) for k in (0, 20) for p in (0.95, 1.0)]
EXTRA = [Sampling(seed=11, temperature=0.2, top_k=0, top_p=0.95),
         Sampling(seed=12, temperature=1.0, top_k=0, min_p=0.05),
         Sampling(seed=13, temperature=0.7, top_k=4096, top_p=1.0), Sampling(seed=14, temperature=1.0, top_k=1)]


def _logits(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((len(POSITIONS), VOCAB), generator=g, device="cuda") * 4.0


def _halves(fn, logits: torch.Tensor, sampling):
    """fn on two ranks, each holding its vocabulary half -> both ranks' results."""

    def rank(r):
        def go(c):
            w = SimpleNamespace(comm=c, world=2, vocab_offset=r * HALF)
            return fn(w, logits[:, r * HALF:(r + 1) * HALF].contiguous(), POSITIONS, sampling)
        return go

    return run_pair(rank(0), rank(1))


def _whole(fn, logits: torch.Tensor, sampling):
    return fn(SimpleNamespace(comm=None, world=1, vocab_offset=0), logits, POSITIONS, sampling)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32, torch.uint8])
def test_all_gather_is_cat_in_rank_order(dtype):
    sends = [(torch.arange(37, device="cuda") + 100 * r).to(dtype) for r in range(2)]

    def rank(r):
        def go(c):
            got = torch.empty(2 * 37, dtype=dtype, device="cuda")
            C.fast_gather(c, sends[r], got)                 # PairComm has no fast path: its all_gather
            return got
        return go

    a, b = run_pair(rank(0), rank(1))
    want = torch.cat(sends)
    assert torch.equal(a, want) and torch.equal(b, want)


def test_run_pair_raises_the_failing_rank_and_frees_the_other():
    def gathers(c):
        x = torch.zeros(4, device="cuda")
        c.all_gather(x, torch.empty(8, device="cuda"))

    def fails(c):
        raise KeyError("rank 1")

    with pytest.raises(KeyError, match="rank 1"):
        run_pair(gathers, fails)
    with pytest.raises(KeyError, match="rank 1"):
        run_pair(fails, gathers)


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=str)
def test_target_rows_on_two_halves_equal_the_full_vocabulary(sampling):
    logits = _logits(sampling.seed if sampling else 1)
    want, host = _whole(S.target_rows, logits, sampling)
    (a, ta), (b, tb) = _halves(S.target_rows, logits, sampling)
    assert a == want and b == want
    assert min(host, ta, tb) >= 0.0


def _host_draft(logits: torch.Tensor, sampling) -> list[int]:
    """The draft rule on the host: the target's keyed rule with top_k capped at, and off read as, ``DRAFT_TOP_K``."""

    if sampling is not None and sampling.temperature > 0:
        sampling = replace(sampling, top_k=min(sampling.top_k or S.DRAFT_TOP_K, S.DRAFT_TOP_K))
    return _whole(S.target_rows, logits, sampling)[0]


def _device_drafts(logits: torch.Tensor, sampling, world: int, sparse=None) -> tuple[list[int], ...]:
    """``S.draft`` of every row on ``world`` ranks, each holding its vocabulary slice, at ``POSITIONS`` (two positions
    ahead of the state's) -> each rank's drafts."""

    width = VOCAB // world

    def rank(r):
        def go(c):
            w = SimpleNamespace(comm=c, world=world, vocab_offset=r * width)
            d = S.Draws(VOCAB, world)
            keyed = d.set(sampling)
            out = torch.zeros(len(POSITIONS), dtype=torch.int32, device="cuda")
            for i, position in enumerate(POSITIONS):
                pos = torch.tensor([position - 2], dtype=torch.int32, device="cuda")
                row = logits[i, r * width:(r + 1) * width].contiguous()
                bias = None if sparse is None else (sparse[0][r * width:(r + 1) * width], sparse[1])
                S.draft(w, d, row, pos, 2, keyed, out[i:i + 1], bias)
            return out.tolist()
        return go

    if world == 1:
        return (rank(0)(None),)
    return run_pair(rank(0), rank(1))


@pytest.mark.parametrize("sampling", SAMPLINGS + EXTRA, ids=str)
def test_device_drafts_follow_the_host_rule_on_one_and_two_ranks(sampling):
    logits = _logits(sampling.seed if sampling else 1)
    want = _host_draft(logits, sampling)
    assert _device_drafts(logits, sampling, 1) == (want,)
    assert _device_drafts(logits, sampling, 2) == (want, want)
    if sampling is None or sampling.temperature <= 0 or 0 < sampling.top_k <= S.DRAFT_TOP_K:
        assert want == _whole(S.target_rows, logits, sampling)[0], "a draft is the target's draw"


@pytest.mark.parametrize("sampling", [None, Sampling(seed=4, temperature=1.0, top_k=0, top_p=0.95),
                                      Sampling(seed=4, temperature=0.2, top_k=20, top_p=1.0)], ids=str)
def test_sparse_drafts_add_the_bias_to_the_base_candidates_only(sampling):
    """Each rank's top candidates by base logit, those gaining the head . me bias, then the host rule over them."""

    logits = _logits(9)
    g = torch.Generator(device="cuda").manual_seed(9)
    head = torch.randn((VOCAB, 64), generator=g, device="cuda").bfloat16()
    me = (torch.randn((64,), generator=g, device="cuda") / 4).bfloat16()
    bias = (head.double() @ me.double()).float()
    width, n = HALF, S.CANDIDATES
    masked = torch.full_like(logits, float("-inf"))
    for r in range(2):
        half = logits[:, r * width:(r + 1) * width]
        cols = half.topk(n, dim=-1).indices + r * width
        masked.scatter_(1, cols, logits.gather(1, cols) + bias[cols])
    want = _host_draft(masked, sampling)
    assert _device_drafts(logits, sampling, 2, (head, me)) == (want, want)
    assert want != _host_draft(logits, sampling), "the bias moved no draft"
