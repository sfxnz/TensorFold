"""The target forward on one rank: embedding, Engram, the mHC blocks, the final norm and head rows."""

from __future__ import annotations

import threading
from collections.abc import Generator, Mapping, Sequence

import numpy as np
import torch

from tensorfold.cuda.comm import fast_gather
from tensorfold.families.glm5_next.cuda import glue

from ..engram_hash import rank_columns
from . import compressor, engram, moe, mx8, norms
from .attention import attention
from .buffers import Buffers, State
from .weights import LayerW, Weights

HALF_ROWS = 640         # two ranks run a prompt chunk of twice this many rows or more as two row halves
_LOCAL = threading.local()


def stage(w: Weights, st: State, b: Buffers, tokens: Sequence[int], hasher=None, reader=None, half: int = 0,
          ready: int = 0) -> int:
    """Host work before a forward of ``tokens``: ids and this rank's Engram rows through pinned halves -> R; the first
    ``ready`` rows' Engram records are in the half already (``engram.fill_rows``)."""

    R = len(tokens)
    if not 0 < R <= b.rows or st.pos + R > st.capacity:
        raise ValueError(f"{R} rows at {st.pos}: buffers hold {b.rows} rows, the state {st.capacity} positions")
    b.staged.synchronize()
    b.ids_host[:R].numpy()[:] = tokens
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    b.staged.record()
    if w.engram:
        ids = np.empty((0, *b.eraw.shape[1:3]), dtype=np.int64)
        if ready < R:
            back = w.cfg.engram_max_ngram_size - 1
            before = (list(st.history) + list(tokens[:ready]))[-back:] if back > 0 else []
            ids = hasher.ids(np.asarray(before, dtype=np.int64), np.asarray(tokens[ready:], dtype=np.int64))
            ids = rank_columns(ids, w.rank, w.world)
        engram.stage_rows(ids, reader, b.eraw_host[half], b.eraw_done[half], b.eraw, scales=w.engram_scales,
                          idx_host=b.eidx_host[half] if b.eidx_host else None, idx=b.eidx, first=ready)
    return R


def _gather(w: Weights, b: Buffers, R: int, start: int = 0, *, trade: bool = False) -> torch.Tensor:
    """Every rank's fp32 share ``b.part[start:R]`` -> [world, R - start, D] in rank order (one rank: the share), in
    the gather slots of rows start..R; ``trade``: two ranks send and receive point to point, which leaves the SMs to
    the kernels beside it (NCCL's all-gather runs a block on every channel it has), unless a transport gathers."""

    part = b.part[start:R]
    if w.world == 1:
        return part[None]
    D = part.shape[-1]
    recv = b.gath.view(-1)[w.world * start * D:w.world * R * D].view(w.world, R - start, D)
    exchange = getattr(w.comm, "exchange", None) if trade and w.world == 2 else None
    if exchange is None or getattr(w.comm, "all_gather_fast", None) is not None:
        fast_gather(w.comm, part, recv)
    else:
        peer = 1 - w.rank
        recv[w.rank].copy_(part)
        exchange([part], [recv[peer]], peer)
    return recv


def _tap(b: Buffers, R: int, slot: int, first: int = 0, lo: int = 0, hi: int | None = None) -> None:
    """bf16(mean of the 4 copies) of the forward's last min(R - first, taps) rows, those among rows lo..hi, into DSpark
    input slot ``slot``: tap row t is forward row R - min(R - first, taps) + t."""

    hi = R if hi is None else hi
    end = R - min(R - first, b.taps.shape[0])
    a = max(end, lo)
    if hi <= a:
        return
    glue.stream_mean(b.X[a:hi].view(hi - a, -1), b.hidden[:hi - a])
    b.taps[a - end:hi - end, slot].copy_(b.hidden[:hi - a])


def _block(lw: LayerW, w: Weights, st: State, b: Buffers, R: int, e: torch.Tensor | None, prompt: bool,
           start: int = 0, kv_from: int = 0, taps_from: int = 0, lo: int = 0,
           hi: int | None = None) -> Generator[int, torch.Tensor, None]:
    """One block (M:968-994) on rows start..hi of the R-row forward ``b.X``, in place, rows lo.. its own (Engram, taps,
    window KV from kv_from); yields each fp32 share's first row and takes the shares gathered. ``b.pre_in``: FFN pre."""

    hi = R if hi is None else hi
    cfg, L = w.cfg, lw.index
    eps = cfg.rms_norm_eps
    if L in w.engram:
        eg = w.engram[L]
        kv = engram.kv(e[lo:hi, cfg.engram_layer_ids.index(L)], eg.wkv, w.comm, b.ekv, b.ekv_gat, prompt=prompt)
        engram.inject(b.X[lo:hi], kv, eg.wqk, eps)
    if lw.role.tap is not None:
        _tap(b, R, lw.role.tap, taps_from, lo, hi)
    if start == hi:                             # window KV and pooling only, or nothing
        if kv_from < hi or (lw.role.ratio and lw.role.kv_src == L):
            attention(lw, w, st, b, hi, prompt, start, kv_from, lo)
        return
    n = hi - start
    flat = b.X[start:hi].view(n, -1)
    h = lw.hc_attn
    norms.hc_mix(flat, h.fn, h.base, h.scale, b.hcpart[start:hi], b.pre_a[start:hi], b.post_a[start:hi],
                 b.comb_a[start:hi], eps, cfg.hc_eps, cfg.hc_sinkhorn_iters)
    attention(lw, w, st, b, hi, prompt, start, kv_from, lo)
    glue.hc_post(flat, flat, (yield start), b.post_a[start:hi], b.comb_a[start:hi])
    h = lw.hc_ffn
    norms.hc_mix(flat, h.fn, h.base, h.scale, b.hcpart[start:hi], b.pre_f[start:hi], b.post_f[start:hi],
                 b.comb_f[start:hi], eps, cfg.hc_eps, cfg.hc_sinkhorn_iters)
    xf = norms.collapse_norm(flat, b.pre_a[start:hi], lw.ffn_norm, b.xn[start:hi], eps)   # delayed mHC: attn's pre
    moe.backbone(cfg, lw.moe, xf, b, prompt=prompt, comm=w.comm, out=b.part[start:hi])
    glue.hc_post(flat, flat, (yield start), b.post_f[start:hi], b.comb_f[start:hi])
    b.pre_in[start:hi].copy_(b.pre_f[start:hi])


def layer(lw: LayerW, w: Weights, st: State, b: Buffers, R: int, e: torch.Tensor | None, prompt: bool,
          start: int = 0, kv_from: int = 0, taps_from: int = 0) -> None:
    """One block on rows start.. of ``b.X[:R]`` in place (window KV from kv_from), each share gathered as made."""

    steps = _block(lw, w, st, b, R, e, prompt, start, kv_from, taps_from)
    try:
        first = next(steps)
        while True:
            first = steps.send(_gather(w, b, R, first))
    except StopIteration:
        pass


def _halves(w: Weights, st: State, b: Buffers, R: int, e: torch.Tensor | None,
            rows: Mapping[int, tuple[int, int]], taps_from: int) -> None:
    """A prompt chunk's rows [0, h) and [h, R) block by block in turn on this stream, each half's shares gathered on a
    second one while the other half computes; the kernels are row-invariant, so the bits are one pass's."""

    h = -(-R // 256) * 128
    main = torch.cuda.current_stream()
    side = vars(_LOCAL).setdefault("streams", {}).get(main.device)
    if side is None:                    # this thread's (two thread ranks on one GPU must not share it)
        side = _LOCAL.streams[main.device] = torch.cuda.Stream(main.device)

    def half(lo: int, hi: int) -> Generator[int, torch.Tensor, None]:
        for lw in w.layers:
            s, k = rows.get(lw.index, (0, 0))
            yield from _block(lw, w, st, b, R, e, True, min(max(s, lo), hi), min(max(k, lo), hi), taps_from, lo, hi)

    steps, ends, got = [half(0, h), half(h, R)], (h, R), [None, None]
    while steps[0] or steps[1]:
        for i in (0, 1):
            if steps[i] is None:
                continue
            if got[i] is not None:
                main.wait_event(got[i][0])
            try:
                first = steps[i].send(None if got[i] is None else got[i][1])
            except StopIteration:
                steps[i] = None
                continue
            side.wait_stream(main)
            with torch.cuda.stream(side):
                recv = _gather(w, b, ends[i], first, trade=True)
                got[i] = (side.record_event(), recv)


def head(w: Weights, b: Buffers, a: int, n: int, out: torch.Tensor, *, prompt: bool) -> torch.Tensor:
    """Rows a..a+n of the last forward: final collapse and norm, then the rank's vocabulary rows, fp32."""

    norms.collapse_norm(b.X[a:a + n].view(n, -1), b.pre_in[a:a + n], w.norm, b.fnormed[:n], w.cfg.rms_norm_eps)
    return mx8.mm(w.head, b.fnormed[:n], out, f32=True, prompt=prompt)


def compute(w: Weights, st: State, b: Buffers, R: int, *, prompt: bool, head_rows: int,
            rows: Mapping[int, tuple[int, int]] | None = None, taps_from: int = 0) -> torch.Tensor | None:
    """Staged rows through every loaded layer -> the last ``head_rows`` rows' fp32 logits; commits nothing.

    ``rows``: a layer's (first row of its block, first row writing window KV), the rows before left as they are.
    """

    if not 0 < R <= b.rows or not 0 <= head_rows <= min(R, b.logits.shape[0]):
        raise ValueError(f"compute: {R} rows, head on {head_rows}, for {b.rows}-row buffers and "
                         f"{b.logits.shape[0]} logit rows")
    cfg = w.cfg
    glue.embed(b.ids[:R], w.embed, cfg.hidden_size, cfg.hc_mult, b.X[:R])
    b.pre_in[:R].zero_()
    b.pre_in[:R, 0].fill_(1.0)                  # the first block collapses to copy 0, the embedding itself
    e = None
    if w.engram:
        e = engram.exchange(engram.dequant_rows(b.eraw[:R], b.eloc[:R], w.engram_scales, b.eidx[:R]), w.comm, b.egat,
                            b.eng)
    if prompt and w.world > 1 and R >= 2 * HALF_ROWS:
        _halves(w, st, b, R, e, rows or {}, taps_from)
    else:
        for lw in w.layers:
            layer(lw, w, st, b, R, e, prompt, *(rows or {}).get(lw.index, (0, 0)), taps_from)
    if not head_rows:
        return None
    return head(w, b, R - head_rows, head_rows, b.logits[:head_rows], prompt=prompt)


@torch.no_grad()
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int, first: int = 0, row0: int = 0) -> None:
    """Keep the last forward's rows first..keep of the R from row ``row0`` (a lane's segment), row first at
    ``st.pos`` (a prompt chunk may keep its rows in two steps): window KV into the rings, the tails, then
    ``pos += keep - first``."""

    if not 0 <= first < keep <= R <= b.rows - row0 or row0 < 0 or (first and not b.prefill):
        raise ValueError(f"commit: rows {first}..{keep} of {R} from {row0} "
                         f"({'a prompt chunk' if b.prefill else 'decode'})")
    win, layers = st.rings.shape[1], w.cfg.num_hidden_layers
    n, at = min(keep - first, win), st.pos - first        # at: row 0's position
    slots = torch.arange(keep - n, keep, device=st.rings.device).add_(at).remainder_(win)
    st.rings[:layers].index_copy_(1, slots, b.kvw[:layers, row0 + keep - n:row0 + keep])
    compressor.commit_tail(st, b, at, keep, row0)
    back = w.cfg.engram_max_ngram_size - 1
    st.history = (st.history + b.ids_host[row0 + first:row0 + keep].tolist())[-back:] if back > 0 else []
    st.set_pos(at + keep)
