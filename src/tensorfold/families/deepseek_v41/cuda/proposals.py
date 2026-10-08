"""DSpark proposals of several lanes in one block forward: each lane's block rows through the stages and the head on its
own ring and positions, then the Markov steps over the lanes; every draft has the bits of the lane's own proposal."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence

import torch
import triton

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.comm import fast_gather
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import glue, qmm

from ..config import Config
from . import attn_kernel, dspark, mx8, norms, sample
from .buffers import Buffers
from .lanes import Lanes

ENV = "TF_DSV41_BATCHED_DRAFTS"     # "1": two or more drafting lanes propose in one block forward
LAST = torch.iinfo(torch.int64).max  # a greedy slot's unused candidate keys: after every real key


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


class Batch:
    """Scratch of one block over up to ``slots`` lanes, slot j on rows j B..: each slot's lane, token and draw knobs
    (``table``, from the host), each row's lane, position, anchor and list, the block's DSpark rows, the draws'
    candidates, and the pinned rows each Markov step's drafts and confidences land in."""

    def __init__(self, cfg: Config, world: int, slots: int, device: torch.device | str = "cuda") -> None:
        dev = torch.device(device)
        bf, f32, i32, i64 = torch.bfloat16, torch.float32, torch.int32, torch.int64
        B, D, V = cfg.dspark_block_size, cfg.hidden_size, cfg.vocab_size // world
        n, DK, E = B * slots, cfg.dspark_num_experts_per_tok + 1, cfg.dspark_n_routed_experts   # the shared expert last
        self.slots, self.block, self.world = slots, B, world
        host = dev.type == "cuda"
        self.table = torch.zeros((slots, 7), dtype=i64, device=dev)       # lane, token, then sample.knobs
        self.table_host = torch.zeros((slots, 7), dtype=i64, pin_memory=host)
        self.staged = torch.cuda.Event() if host else None
        self.lane = torch.zeros((slots,), dtype=i32, device=dev)
        self.at = torch.zeros((slots,), dtype=i32, device=dev)            # each slot's lane position
        rows = torch.arange(n, dtype=i64, device=dev)
        self.slot, self.step = rows // B, (rows % B).to(i32)
        self.rlane = torch.zeros((n,), dtype=i32, device=dev)
        self.base = torch.zeros((n,), dtype=i32, device=dev)
        self.anchors = torch.zeros((n,), dtype=i32, device=dev)          # p for every block row (M:1021-1029)
        self.rpos = torch.zeros((n,), dtype=i32, device=dev)
        self.lists = (self.slot * B)[:, None].add(torch.arange(B, dtype=i64, device=dev)).to(i32)   # its slot's rows
        self.counts = torch.full((n,), B, dtype=i32, device=dev)
        self.ids = torch.full((n,), cfg.dspark_noise_token_id, dtype=i32, device=dev)
        self.chain = torch.zeros((B + 1, slots), dtype=i32, device=dev)  # step-major: chain[i] the slots' inputs i
        self.xd, self.bkv = torch.empty((n, cfg.hc_mult, D), dtype=bf, device=dev), torch.empty(
            (n, cfg.head_dim), dtype=bf, device=dev)
        self.dmlog = torch.empty((n, E), dtype=f32, device=dev)
        self.dpick, self.dwts = torch.empty((n, DK), dtype=i32, device=dev), torch.empty((n, DK), dtype=f32, device=dev)
        self.dplan = grouped.Plan(n, DK, E + 1, dev)
        self.dact = torch.empty((n, DK, cfg.moe_intermediate_size // world), dtype=bf, device=dev)
        self.dey = torch.empty((n, DK, D), dtype=f32, device=dev)
        self.dlog = torch.empty((n, V), dtype=f32, device=dev)
        self.me = torch.empty((B, slots, cfg.dspark_markov_rank), dtype=bf, device=dev)
        self.conf = torch.empty((B, slots), dtype=f32, device=dev)
        self.bias = torch.empty((slots, V), dtype=f32, device=dev)
        c = self.width = min(sample.CANDIDATES, V)
        self.vals = torch.empty((slots, c), dtype=f32, device=dev)
        self.cols = torch.zeros((slots, c), dtype=i64, device=dev)
        self.keys = torch.zeros((slots, c), dtype=i64, device=dev)
        self.got = torch.zeros((world * slots * c,), dtype=i64, device=dev)
        self.cand = torch.zeros((slots, world * c), dtype=i64, device=dev)
        self.order = torch.empty((slots, world * c), dtype=i64, device=dev)
        self.idx = torch.empty((slots, world * c), dtype=i64, device=dev)
        self.host = torch.empty((B, 2, slots), dtype=i32, pin_memory=host)    # each step's drafts, confidence bits
        self.landed = [torch.cuda.Event(external=True) for _ in range(B)]
        self.waited = self.launch = 0.0         # host seconds the last proposal waited for steps and launched


def block(w, lanes: Lanes, b: Buffers, k: Batch, n: int) -> None:
    """The three stages, collapse and norm of block rows 0..n (M:1128-1146), then the head into ``k.dlog``."""

    cfg, ds, L = w.cfg, w.dspark, n // k.block
    torch.index_select(k.lane, 0, k.slot[:n], out=k.rlane[:n])
    torch.index_select(lanes.pos_dev, 0, k.rlane[:n], out=k.base[:n])
    torch.sub(k.base[:n], 1, out=k.anchors[:n])
    torch.add(k.base[:n], k.step[:n], out=k.rpos[:n])
    k.ids.view(k.slots, k.block)[:L, 0].copy_(k.chain[0, :L])
    glue.embed(k.ids[:n], w.embed, cfg.hidden_size, cfg.hc_mult, k.xd[:n])
    b.pre_in[:n].zero_()
    b.pre_in[:n, 0].fill_(1.0)
    tables = attn_kernel.LaneArgs(k.rlane, k.rlane, lanes.pos_dev, lanes.rings.stride(0), 0)

    def attend(sw, q: torch.Tensor, kv: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return attn_kernel.attention(q, lanes.rings[0, sw.index], None, lanes.pos_dev, k.anchors[:n], kv,
                                     k.lists[:n], k.counts[:n], sw.attn.sink, out, prompt=False, part=b.attn_part,
                                     lanes=tables)

    for sw in ds.stages:
        dspark.stage(sw, w, b, k.xd[:n], k.rpos[:n], k.bkv[:n], k, attend)
    x = norms.collapse_norm(k.xd[:n].view(n, -1), b.pre_in[:n], ds.norm, b.fnormed[:n], cfg.rms_norm_eps,
                            raw=b.hidden[:n])
    mx8.mm(w.head, x, k.dlog[:n], f32=True)


def _draws(w, k: Batch, i: int, L: int, keyed: int) -> None:
    """Step i's draft of slots 0..L into ``chain[i + 1]``, ``sample.draft``'s rule a slot, the last ``keyed`` slots
    keyed: every slot's candidate keys gathered at once, each slot's sorted alone."""

    B, c = k.block, k.width
    for j in range(L):
        row = k.dlog[j * B + i]
        if j >= L - keyed:
            torch.topk(row, c, sorted=False, out=(k.vals[j], k.cols[j]))
            n = c
        else:
            torch.argmax(row, dim=0, keepdim=True, out=k.cols[j, :1])     # the first of equal maxima: the lowest id
            k.keys[j, 1:].fill_(LAST)
            n = 1
        sample._keys[(triton.cdiv(n, 64),)](row, k.cols[j], row, row, k.keys[j], w.vocab_offset, n, RK=1, BIAS=False,
                                            B=64, num_warps=4)
    cand = k.keys[:L]
    if w.world > 1:
        got = k.got[:w.world * L * c].view(w.world, L, c)
        fast_gather(w.comm, cand, got)
        cand = k.cand[:L]
        cand.view(L, w.world, c).copy_(got.transpose(0, 1))
    torch.sort(cand, dim=-1, out=(k.order[:L], k.idx[:L]))      # keys are distinct: the sort is each slot's own
    for j in range(L):
        out, pos = k.chain[i + 1, j:j + 1], k.at[j:j + 1]
        if j >= L - keyed:
            sample._draw[(1,)](k.order[j], w.world * c, k.table[j, 2:], pos, out, 1 + i, KEYED=True,
                               B=sample.DRAFT_TOP_K, G=64, num_warps=8)
        else:
            sample._draw[(1,)](k.order[j], 1, k.table[j, 2:], pos, out, 1 + i, KEYED=False, B=sample.DRAFT_TOP_K,
                               G=64, num_warps=1)


@torch.no_grad()
def chain(w, lanes: Lanes, b: Buffers, k: Batch, L: int, keyed: int, d: int) -> None:
    """The block of slots 0..L, then ``d`` Markov steps over them, the last ``keyed`` slots drawing keyed; each step's
    drafts and confidences copied to the pinned rows and its event recorded: one graph per (L, keyed, d)."""

    B = k.block
    k.lane[:L].copy_(k.table[:L, 0])
    k.chain[0, :L].copy_(k.table[:L, 1])
    torch.index_select(lanes.pos_dev, 0, k.lane[:L], out=k.at[:L])
    block(w, lanes, b, k, L * B)
    D, RK = w.cfg.hidden_size, w.cfg.dspark_markov_rank
    for i in range(d):
        dspark._markov_in[(L,)](k.chain, w.dspark.markov_embed, b.hidden, w.dspark.conf, k.me, k.conf, i, k.slots, B,
                                D=D, RK=RK, BLOCK=triton.next_power_of_2(D + RK), num_warps=8)
        qmm.matmul(k.me[i, :L], w.dspark.markov_head, out=k.bias[:L], f32=True)
        k.dlog.view(k.slots, B, -1)[:L, i].add_(k.bias[:L])
        _draws(w, k, i, L, keyed)
        k.host[i, 0, :L].copy_(k.chain[i + 1, :L], non_blocking=True)
        k.host[i, 1, :L].copy_(k.conf[i, :L].view(torch.int32), non_blocking=True)
        k.landed[i].record()


def set_slots(k: Batch, slots: Sequence[tuple[int, int, Sampling | None]]) -> int:
    """Each slot's (lane, token, sampling) into ``table`` through its pinned twin -> how many slots draw keyed, the
    last ones; greedy slots must come first."""

    flags = [sample.keyed(s) for _, _, s in slots]
    if flags != sorted(flags) or not 0 < len(slots) <= k.slots:
        raise ValueError(f"{len(slots)} slots of {k.slots}, keyed {flags}: greedy slots first")
    k.staged.synchronize()
    host = k.table_host.numpy()
    host[:] = 0
    for j, (lane, y, s) in enumerate(slots):
        host[j, :2] = lane, y
        if flags[j]:
            host[j, 2:] = sample.knobs(s, k.world * k.width)
    k.table.copy_(k.table_host, non_blocking=True)
    k.staged.record()
    return sum(flags)


@torch.no_grad()
def propose(k: Batch, wanted: Sequence[tuple[int, int, Sampling | None, int]], conf_threshold: float | None, *,
            run: Callable[[int, int, int], None], landed=None) -> list[tuple[list[int], list[float]]]:
    """Every (lane, token, sampling, d) of ``wanted``'s proposal in one block (``run(L, keyed, d)``: ``chain`` or its
    graph) -> each one's drafts the policy keeps and their confidence logits, as ``dspark.propose`` gives them alone;
    ``landed(index, i, draft)`` as each one lands."""

    if not 1 < len(wanted) <= k.slots or any(not 1 <= d <= k.block for *_, d in wanted):
        raise ValueError(f"propose: {len(wanted)} lanes of {k.slots}, drafts {[d for *_, d in wanted]}")
    order = sorted(range(len(wanted)), key=lambda x: sample.keyed(wanted[x][2]))
    keyed = set_slots(k, [wanted[x][:3] for x in order])
    run(len(order), keyed, max(d for *_, d in wanted))
    drafts: list[list[int]] = [[] for _ in wanted]
    conf: list[list[float]] = [[] for _ in wanted]
    live = list(enumerate(order))
    k.waited = 0.0
    for i in range(k.block):
        live = [(j, x) for j, x in live if i < wanted[x][3]]
        if not live:
            break
        start = time.perf_counter()
        k.landed[i].synchronize()
        k.waited += time.perf_counter() - start
        going = []
        for j, x in live:
            conf[x].append(float(k.host[i, 1, j:j + 1].view(torch.float32)))
            if dspark.policy(conf[x], wanted[x][3], conf_threshold) <= i:
                continue
            drafts[x].append(int(k.host[i, 0, j]))
            if landed is not None:
                landed(x, i, drafts[x][-1])
            going.append((j, x))
        live = going
    return [(dr, c[:len(dr)]) for dr, c in zip(drafts, conf)]
