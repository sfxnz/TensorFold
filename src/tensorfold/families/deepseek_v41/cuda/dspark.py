"""DSpark drafting on one rank: stage rings, a 5-row block through three stages, then Markov draws."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import torch
import triton
import triton.language as tl

from tensorfold.cuda.comm import fast_gather
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import glue, qmm

from ..config import Config
from . import attn_kernel, moe, mx8, norms, quant, rope, sample
from .buffers import Buffers
from .forward import _gather
from .weights import StageW, Weights

SPARSE_MARKOV = False       # the Markov bias on each rank's top base candidates only, not every vocabulary row


class Work:
    """The drafter's own device scratch: block ids, the Markov chain's tokens, the block rows' anchors and lists, a
    bias row, the draws' scratch and knobs, and the pinned rows the drafts and confidences come back in."""

    def __init__(self, cfg: Config, world: int, device: torch.device | str = "cuda",
                 sparse: bool | None = None) -> None:
        dev, i32 = torch.device(device), torch.int32
        B, win = cfg.dspark_block_size, cfg.sliding_window
        self.bids = torch.full((B,), cfg.dspark_noise_token_id, dtype=i32, device=dev)    # [y, noise, ...]
        self.chain = torch.zeros((B + 1,), dtype=i32, device=dev)    # out_0 = y, then out_{i+1} = draft i
        self.mids = self.chain[:B]                                    # out_i: Markov step i's input token
        self.anchors = torch.zeros((B,), dtype=i32, device=dev)       # p for every block row (M:1021-1029)
        self.lists = torch.arange(B, dtype=i32, device=dev).repeat(B, 1)    # every row sees all block rows
        self.counts = torch.full((B,), B, dtype=i32, device=dev)
        self.bias = torch.empty((1, cfg.vocab_size // world), dtype=torch.float32, device=dev)
        self.rows = torch.arange(win, dtype=torch.int64, device=dev)
        self.at = torch.empty((win,), dtype=torch.int64, device=dev)   # absorbed positions
        self.sparse = SPARSE_MARKOV if sparse is None else sparse
        self.draws = sample.Draws(cfg.vocab_size, world, dev)
        self.host = torch.empty((2, B), dtype=i32, pin_memory=dev.type == "cuda")     # drafts, confidence bits
        self.landed = [torch.cuda.Event(external=True) for _ in range(B)]   # step i's row reached the host
        self.waited = 0.0                                                   # host seconds the last propose waited


def main_rows(w, b: Buffers, first: int, n: int, *, prompt: bool) -> torch.Tensor:
    """``main_proj`` of tap rows first.. over the ranks, then ``main_norm`` -> ``b.xn[:n]`` (M:1039-1043)."""

    ds, x = w.dspark, b.mx[:n]
    taps = b.taps[first:first + n].view(n, -1)
    if w.world == 1:
        mx8.mm(ds.main_proj, taps, x, prompt=prompt)
    else:                               # the rank's output columns, gathered and laid side by side in rank order
        part = b.mx.view(-1)[:x.numel() // w.world].view(n, -1)
        mx8.mm(ds.main_proj, taps, part, prompt=prompt)
        recv = b.mx_gat.view(-1)[:x.numel()].view(w.world, n, -1)
        fast_gather(w.comm, part, recv)
        x.view(n, w.world, -1).copy_(recv.transpose(0, 1))
    return norms.rmsnorm(x, ds.main_norm, w.cfg.rms_norm_eps, b.xn[:n])


@torch.no_grad()
def absorb(e, b: Buffers, n: int, *, prompt: bool, first: int = 0) -> None:
    """The last ``n`` committed positions' taps, tap rows first.., into every stage's ring (M:1039-1051, 1128-1130)."""

    w, st, k = e.w, e.st, e.dwork
    cfg, ds = w.cfg, w.dspark
    if not 0 < n <= min(b.taps.shape[0] - first, st.pos, st.rings.shape[1]) or first < 0:
        raise ValueError(f"absorb: tap rows {first}..{first + n} at {st.pos} committed positions, "
                         f"{b.taps.shape[0]} tap rows, a {st.rings.shape[1]}-slot ring")
    eps = cfg.rms_norm_eps
    main_x = main_rows(w, b, first, n, prompt=prompt)
    at = torch.add(k.rows[:n], st.pos_dev, out=k.at[:n]).sub_(n)
    slots = at.remainder(st.rings.shape[1])
    for sw in ds.stages:
        qakv = mx8.mm(sw.attn.wqa_kv, main_x, b.qakv[:n], prompt=prompt)
        kv = quant.norm_rope_fp8(qakv[:, cfg.q_lora_rank:], sw.attn.kv_norm, eps, at, w.rope[sw.role.rope],
                                 b.kvw[sw.index, :n])
        st.rings[sw.index].index_copy_(0, slots, kv)


def stage(sw: StageW, w: Weights, b: Buffers, xd: torch.Tensor, at: torch.Tensor, kv_out: torch.Tensor, d,
          attend: Callable[..., torch.Tensor]) -> None:
    """One stage (M:968-994, attention as M:1054-1074) on the block rows ``xd`` in place at positions ``at``: their KV
    into ``kv_out``, ``attend(sw, q, kv, out)`` the attention, ``d`` the scratch of the DSpark MoE's rows."""

    cfg, a = w.cfg, sw.attn
    n, eps, table = xd.shape[0], cfg.rms_norm_eps, w.rope[sw.role.rope]
    flat = xd.view(n, -1)
    h = sw.hc_attn
    norms.hc_mix(flat, h.fn, h.base, h.scale, b.hcpart[:n], b.pre_a[:n], b.post_a[:n], b.comb_a[:n], eps,
                 cfg.hc_eps, cfg.hc_sinkhorn_iters)
    xa = norms.collapse_norm(flat, b.pre_in[:n], sw.attn_norm, b.xn[:n], eps)
    qakv = mx8.mm(a.wqa_kv, xa, b.qakv[:n])
    qr = norms.rmsnorm(qakv[:, :cfg.q_lora_rank], a.q_norm, eps, b.qr[:n])
    kv = quant.norm_rope_fp8(qakv[:, cfg.q_lora_rank:], a.kv_norm, eps, at, table, kv_out)
    q = b.q[:n]
    mx8.mm(a.wq_b, qr, q.view(n, -1))
    rope.apply(q, at, table)
    o = attend(sw, q, kv, b.o[:n])
    rope.apply(o, at, table, inverse=True)
    u = mx8.grouped(a.wo_a, o.view(n, -1), b.u[:n])
    mx8.mm(a.wo_b, u, b.part[:n], f32=True)
    glue.hc_post(flat, flat, _gather(w, b, n), b.post_a[:n], b.comb_a[:n])
    h = sw.hc_ffn
    norms.hc_mix(flat, h.fn, h.base, h.scale, b.hcpart[:n], b.pre_f[:n], b.post_f[:n], b.comb_f[:n], eps,
                 cfg.hc_eps, cfg.hc_sinkhorn_iters)
    xf = norms.collapse_norm(flat, b.pre_a[:n], sw.ffn_norm, b.xn[:n], eps)    # delayed mHC: the attention's pre
    moe.dspark_moe(cfg, sw.moe, xf, b, d)
    glue.hc_post(flat, flat, _gather(w, b, n), b.post_f[:n], b.comb_f[:n])
    b.pre_in[:n].copy_(b.pre_f[:n])


@torch.no_grad()
def block(e) -> torch.Tensor:
    """The three stages, collapse and norm for ``dwork.bids`` (M:1128-1146), then the head into ``dbuf.dlog``."""

    w, st, b, k = e.w, e.st, e.dbuf, e.dwork
    cfg, ds = w.cfg, w.dspark
    B = cfg.dspark_block_size
    glue.embed(k.bids, w.embed, cfg.hidden_size, cfg.hc_mult, b.xd)
    b.pre_in[:B].zero_()
    b.pre_in[:B, 0].fill_(1.0)
    torch.sub(st.pos_dev.expand(B), 1, out=k.anchors)

    def attend(sw: StageW, q: torch.Tensor, kv: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return attn_kernel.attention(q, st.rings[sw.index], None, st.pos_dev, k.anchors, kv, k.lists, k.counts,
                                     sw.attn.sink, out, prompt=False, part=b.attn_part)

    for sw in ds.stages:
        stage(sw, w, b, b.xd, st.pos_dev, b.bkv, b, attend)
    x = norms.collapse_norm(b.xd.view(B, -1), b.pre_in[:B], ds.norm, b.fnormed[:B], cfg.rms_norm_eps,
                            raw=b.hidden[:B])
    return mx8.mm(w.head, x, b.dlog, f32=True)


@triton.jit
def _markov_in(IDS, EMB, X, PROJ, ME, CONF, i, step, xlane, D: tl.constexpr, RK: tl.constexpr, BLOCK: tl.constexpr):
    """Program j, step i: r = i step + j; me[r] = markov_embed[ids[r]] (a bf16 copy); conf[r] = proj . f32(cat(x, me))
    as one fp32 sum, x the row i + j xlane."""

    j = tl.program_id(0)
    r = i * step + j
    n = tl.arange(0, BLOCK)
    tok = tl.load(IDS + r).to(tl.int64)
    isx = n < D
    ism = (n >= D) & (n < D + RK)
    me = tl.load(EMB + tok * RK + (n - D), mask=ism, other=0.0)
    tl.store(ME + r * RK + (n - D), me, mask=ism)
    x = tl.load(X + (i + j * xlane) * D + n, mask=isx, other=0.0)
    v = tl.where(isx, x.to(tl.float32), me.to(tl.float32))
    tl.store(CONF + r, tl.sum(v * tl.load(PROJ + n, mask=n < D + RK, other=0.0), axis=0))


def markov_input(e, i: int) -> None:
    """Row i's Markov embedding into ``dbuf.me[i]`` and confidence logit into ``dbuf.conf[i]`` (M:1149-1156)."""

    w, b, k = e.w, e.dbuf, e.dwork
    D, RK = w.cfg.hidden_size, w.cfg.dspark_markov_rank
    _markov_in[(1,)](k.mids, w.dspark.markov_embed, b.hidden, w.dspark.conf, b.me, b.conf, i, 1, 0, D=D, RK=RK,
                     BLOCK=triton.next_power_of_2(D + RK), num_warps=8)


def markov_step(e, i: int, keyed: bool) -> None:
    """Markov step i: draft i from ``dbuf.dlog[i]`` plus the ``markov_head(me_i)`` bias (fp32, every row of the rank's
    vocabulary, or with ``sparse`` its candidates only), drawn on the device at position p + 2 + i into ``chain``."""

    w, st, b, k = e.w, e.st, e.dbuf, e.dwork
    row, out = b.dlog[i], k.chain[i + 1:i + 2]
    if k.sparse:
        sample.draft(w, k.draws, row, st.pos_dev, 1 + i, keyed, out, (w.dspark.markov_head.weight, b.me[i]))
        return
    qmm.matmul(b.me[i:i + 1], w.dspark.markov_head, out=k.bias, f32=True)
    row.add_(k.bias[0])
    sample.draft(w, k.draws, row, st.pos_dev, 1 + i, keyed, out)


@torch.no_grad()
def chain(e, d: int, keyed: bool) -> None:
    """The block, then ``d`` Markov steps, each draft and confidence copied to the pinned rows and its event recorded:
    the device work of a proposal, one graph per (d, keyed)."""

    b, k = e.dbuf, e.dwork
    block(e)
    for i in range(d):
        markov_input(e, i)
        markov_step(e, i, keyed)
        k.host[0, i:i + 1].copy_(k.chain[i + 1:i + 2], non_blocking=True)
        k.host[1, i:i + 1].copy_(b.conf[i:i + 1].view(torch.int32), non_blocking=True)
        k.landed[i].record()


def _sigmoid(c: float) -> float:
    return 1.0 / (1.0 + math.exp(-c)) if c >= 0 else math.exp(c) / (1.0 + math.exp(c))


def policy(confidence: Sequence[float], d_fixed: int, conf_threshold: float | None = None) -> int:
    """Drafts a round: ``d_fixed``, or the most whose sigmoid product is >= ``conf_threshold``, at least 1."""

    if conf_threshold is None:
        return d_fixed
    n, prod = min(d_fixed, len(confidence)), 1.0
    for i, c in enumerate(confidence[:n]):
        prod *= _sigmoid(float(c))
        if prod < conf_threshold:
            return max(i, 1)
    return n


@torch.no_grad()
def propose(e, y: int, p: int, sampling: Sampling | None, d: int, conf_threshold: float | None = None, *,
            run=None, landed=None) -> tuple[list[int], list[float]]:
    """The block for ``y`` after position ``p`` and ``d`` Markov draws (``run``: ``chain`` or its graph) -> the drafts
    the policy keeps and their confidence logits, alike on every rank; ``landed(i, draft)`` as each one lands."""

    w, st, k = e.w, e.st, e.dwork
    B = w.cfg.dspark_block_size
    if p != st.pos - 1 or p < 0 or not 1 <= d <= B:
        raise ValueError(f"propose: p {p} with {st.pos} committed positions, {d} drafts of a {B}-row block")
    keyed = k.draws.set(sampling)
    k.bids[:1].fill_(y)
    k.chain[:1].fill_(y)
    (run or chain)(e, d, keyed)
    drafts: list[int] = []
    conf: list[float] = []
    k.waited = 0.0
    for i in range(d):
        start = time.perf_counter()
        k.landed[i].synchronize()
        k.waited += time.perf_counter() - start
        conf.append(float(k.host[1, i:i + 1].view(torch.float32)))
        if policy(conf, d, conf_threshold) <= i:
            break
        drafts.append(int(k.host[0, i]))
        if landed is not None:
            landed(i, drafts[-1])
    return drafts, conf[:len(drafts)]
