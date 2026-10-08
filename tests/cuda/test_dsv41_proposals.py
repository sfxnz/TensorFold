"""Proposals of several lanes in one block on the tiny checkpoint: every slot's block logits, hidden rows, Markov inputs,
confidences and drafts equal its lane's own proposal bit for bit at 2, 3 and 4 lanes in any lane order, greedy, keyed
and mixed; one graph replays them for any lanes and tokens; and the DSpark MoE's rows ignore the block's row count."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_tiny
from dsv41_ref_weights import RefWeights

from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import BLOCK, MAX_ROWS, PREFILL_ROWS, dspark, loader, moe, proposals
from tensorfold.families.deepseek_v41.cuda import decode as D
from tensorfold.families.deepseek_v41.cuda import multi as M
from tensorfold.families.deepseek_v41.cuda.buffers import Buffers
from tensorfold.families.deepseek_v41.cuda.lanes import Ahead, Lanes

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
CAP = 1024
S = 4                                   # lanes
PROMPTS = (5, 100, 600, 300)            # the lanes sit at different positions, one past the ring's wrap
K = (Sampling(seed=5, temperature=1.0, top_k=0, top_p=1.0), Sampling(seed=9, temperature=0.8, top_k=20, top_p=0.95),
     Sampling(seed=11, temperature=1.0, top_k=0, top_p=0.9, min_p=0.05))
# slots of one block, greedy first, lanes in any order: (lane, token, sampling)
CASES = [[(3, 17, None), (1, 23, None)], [(0, 29, K[0]), (2, 31, K[1])], [(2, 37, None), (0, 41, K[2])],
         [(1, 43, None), (3, 47, K[0]), (0, 53, K[1])], [(2, 59, K[1]), (1, 61, K[0]), (3, 67, K[2])],
         [(3, 71, None), (2, 73, None), (1, 79, K[0]), (0, 83, K[1])], [(k, 89 + k, None) for k in range(S)],
         [(k, 97 + k, K[k % 3]) for k in (2, 0, 3, 1)]]
OUTS = ("dlog", "hidden", "me", "conf", "drafts")


@pytest.fixture(scope="module")
def ref(tiny_dir):
    return RefWeights(tiny_dir)


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    return loader.load(tiny_dir, Config.read(tiny_dir), 0, 1, None, capacity=CAP)


@pytest.fixture(scope="module")
def rig(tiny, ref):
    """S lanes mid-decode, each prompt filled and some rounds drafted through the lane decoder."""

    cfg, dev = tiny.cfg, tiny.device
    lanes = Lanes(cfg, S, CAP, dev)
    pbuf = Buffers(cfg, 1, min(PREFILL_ROWS, CAP), CAP, prefill=True, device=dev)
    mbuf = Buffers(cfg, 1, MAX_ROWS * S, CAP, device=dev, lanes=S)
    engines = [D.Engine(tiny, CAP, hasher=ref.hasher, reader=ref.reader, st=lanes.view(k), pbuf=pbuf, dbuf=mbuf,
                        dwork=dspark.Work(cfg, 1, dev), ahead=Ahead(cfg, 1)) for k in range(S)]
    dec = M.LaneDecoder(tiny, lanes, engines, mbuf, pbuf, (BLOCK, None), (cfg.eos_token_id,), None, 0.5)
    g = torch.Generator().manual_seed(3)
    for n in PROMPTS:
        dec.admit(Stream(torch.randint(2, cfg.vocab_size, (n,), generator=g).tolist(), 200, None, stop_eos=False))
    while len(dec.streams) < S or min(len(s.out) for s in dec.streams.values()) < 12:
        assert not dec.round(), "no stream ends before the proposals are compared"
    assert len({e.st.pos for e in engines}) == S
    return lanes, mbuf, engines


@pytest.fixture(scope="module")
def batch(tiny):
    return proposals.Batch(tiny.cfg, 1, S, tiny.device)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape).copy_(x).view(torch.uint8)      # a fresh copy: a 1-element column keeps its stride


def _solo(e: D.Engine, y: int, sampling: Sampling | None, d: int) -> dict:
    """Lane ``e``'s own proposal of ``d`` drafts after ``y``, eager -> its rows."""

    k, b = e.dwork, e.dbuf
    keyed = k.draws.set(sampling)
    k.bids[:1].fill_(y)
    k.chain[:1].fill_(y)
    dspark.chain(e, d, keyed)
    torch.cuda.synchronize()
    return {"dlog": b.dlog[:BLOCK].clone(), "hidden": b.hidden[:BLOCK].clone(), "me": b.me[:d].clone(),
            "conf": b.conf[:d].clone(), "drafts": k.chain[1:1 + d].clone()}


def _together(w, rig, k: proposals.Batch, slots, d: int, run=None) -> list[dict]:
    """``slots``' proposals in one block (``run``: a graph's replay, else eager) -> each slot's rows."""

    lanes, b, _ = rig
    keyed = proposals.set_slots(k, slots)
    if run is None:
        proposals.chain(w, lanes, b, k, len(slots), keyed, d)
    else:
        run()
    torch.cuda.synchronize()
    rows = [slice(j * BLOCK, (j + 1) * BLOCK) for j in range(len(slots))]
    return [{"dlog": k.dlog[r].clone(), "hidden": b.hidden[r].clone(), "me": k.me[:d, j].clone(),
             "conf": k.conf[:d, j].clone(), "drafts": k.chain[1:1 + d, j].clone()} for j, r in enumerate(rows)]


def _same(got: dict, want: dict, what: str) -> None:
    for name in OUTS:
        assert torch.equal(_bits(got[name]), _bits(want[name])), f"{what}: {name}"


@pytest.mark.parametrize("case", range(len(CASES)))
def test_every_slot_equals_its_lanes_own_proposal(tiny, rig, batch, case):
    slots, engines = CASES[case], rig[2]
    for d in (1, BLOCK):
        got = _together(tiny, rig, batch, slots, d)
        for j, (lane, y, s) in enumerate(slots):
            _same(got[j], _solo(engines[lane], y, s, d), f"slot {j} (lane {lane}) of {slots}, {d} drafts")
        assert torch.equal(batch.host[:d, 0, :len(slots)], batch.chain[1:1 + d, :len(slots)].cpu()), "the pinned drafts"


def _capture(fn) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


@pytest.mark.parametrize("case", [1, 2, 3, 4, 5, 6])
def test_a_graph_replays_any_lanes_and_tokens(tiny, rig, batch, case):
    lanes, b, engines = rig
    first = CASES[case]
    L, keyed = len(first), sum(s is not None for *_, s in first)
    other = [(lane, y + 1, s) for lane, y, s in first[:L - keyed][::-1] + first[L - keyed:][::-1]]
    proposals.set_slots(batch, first)
    graph = _capture(lambda: proposals.chain(tiny, lanes, b, batch, L, keyed, BLOCK))
    for slots in (first, other):
        eager = _together(tiny, rig, batch, slots, BLOCK)
        for t in (batch.dlog, batch.me, batch.conf, b.hidden):
            t.fill_(float("nan"))
        batch.chain[1:].fill_(-1)
        got = _together(tiny, rig, batch, slots, BLOCK, run=graph.replay)
        for j, (lane, y, s) in enumerate(slots):
            _same(got[j], eager[j], f"replayed slot {j} of {slots}")
            _same(got[j], _solo(engines[lane], y, s, BLOCK), f"replayed slot {j} of {slots} against its lane")


def test_dspark_moe_rows_do_not_depend_on_the_row_count(tiny, rig, batch):
    _, b, _ = rig
    cfg, m, n = tiny.cfg, tiny.dspark.stages[0].moe, BLOCK * S
    g = torch.Generator(device="cuda").manual_seed(4)
    x = (torch.randn((n, cfg.hidden_size), generator=g, device="cuda") * 0.5).to(torch.bfloat16)

    def run(rows: torch.Tensor, d) -> tuple[torch.Tensor, ...]:
        R = rows.shape[0]
        return moe.dspark_moe(cfg, m, rows, b, d).clone(), d.dpick[:R].clone(), d.dwts[:R].clone()

    alone = [torch.cat(t) for t in zip(*(run(x[r:r + 1], b) for r in range(n)))]
    for r in range(1, n + 1):
        for a in (0, n - r):
            for u, v in zip(run(x[a:a + r], batch), alone):
                assert torch.equal(u, v[a:a + r]), (r, a)
    for a in range(0, n, BLOCK):                       # a lane's block through the shared buffers' own scratch
        assert torch.equal(run(x[a:a + BLOCK], b)[0], alone[0][a:a + BLOCK]), a
