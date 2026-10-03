"""Engram on the device: file rows staged through pinned halves, N10a dequant, the column exchange, ``wkv``, N10b.

``M:n`` cites the checkpoint's ``inference/model.py``. Each rank reads its own hash columns of both layers (rank 0
the first half); one gather puts every rank's dequantized rows side by side in column order, the flatten order of
M:353. Each rank projects them through its rows of ``wkv``, and a second gather hands every rank the whole key and
value. Bytes are the file's, the dequant is exact, the gathers copy, and every kernel is per row.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorfold.cuda.comm import fast_gather
from tensorfold.cuda.nvfp4.linear import Mx8Linear

from . import mx8

CLAMP = 1e-6            # M:341: the gate's floor on |dot| before its square root


def stage_rows(ids: np.ndarray, reader, host: torch.Tensor, done: torch.cuda.Event, eraw: torch.Tensor) -> int:
    """E2: rows ``ids`` (int [R, layers, columns], each layer's own row ids) from the files into ``eraw`` -> R.

    ``host`` is one pinned half (u8 [rows, layers, columns, row bytes]) and ``done`` its event: the host writes the
    half only once its previous copy to the device has finished, and ``done`` then marks this copy.
    """

    ids = np.asarray(ids)
    if ids.ndim != 3 or ids.shape[1:] != tuple(host.shape[1:3]) or ids.shape[0] > min(host.shape[0], eraw.shape[0]):
        raise ValueError(f"Engram ids {ids.shape} do not fit staging {tuple(host.shape)} -> {tuple(eraw.shape)}")
    R, layers = ids.shape[0], ids.shape[1]
    starts = np.asarray(reader.layout.starts[:layers], dtype=np.int64)
    flat = (ids.astype(np.int64) + starts[None, :, None]).reshape(-1)
    done.synchronize()
    rec = host.numpy().reshape(-1, host.shape[-1])[:flat.size]
    reader.gather(flat, rec[:, :reader.wrow], rec[:, reader.wrow:])
    eraw[:R].copy_(host[:R], non_blocking=True)
    done.record()
    return R


@triton.jit
def _dequant(RAW, OUT, W: tl.constexpr, S: tl.constexpr):
    """One file row: W e4m3 bytes then S E8M0 bytes -> bf16(e4m3 * 2^(e - 127)) per 32 (0xFF NaN, 0 the subnormal)."""

    p = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, W)
    v = tl.load(RAW + p * (W + S) + j).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    e = tl.load(RAW + p * (W + S) + W + j // (W // S)).to(tl.int32)
    s = tl.where(e == 0, 1 << 22, tl.where(e == 255, 0x7FC00000, e << 23)).to(tl.float32, bitcast=True)
    tl.store(OUT + p * W + j, (v * s).to(tl.bfloat16))


def dequant_rows(eraw: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """N10a: u8 [R, layers, columns, W + W/32] -> out bf16 [R, layers, columns * W], exactly M:312-320's rows."""

    R, layers, cols, width = eraw.shape
    w = width * 32 // 33
    if eraw.dtype != torch.uint8 or not eraw.is_contiguous() or w + w // 32 != width:
        raise ValueError(f"dequant_rows: eraw u8 [R, layers, columns, W + W/32], not {eraw.dtype} {tuple(eraw.shape)}")
    if out.dtype != torch.bfloat16 or not out.is_contiguous() or tuple(out.shape) != (R, layers, cols * w):
        raise ValueError(f"dequant_rows: out bf16 [{R}, {layers}, {cols * w}], not {out.dtype} {tuple(out.shape)}")
    if R:
        _dequant[(R * layers * cols,)](eraw, out, W=w, S=w // 32, num_warps=2)
    return out


def _lead(buf: torch.Tensor, R: int) -> torch.Tensor:
    """[world, R, ...] laid out contiguously at the start of ``buf``'s storage ([world, rows, ...], contiguous)."""

    return buf.view(-1)[:buf.shape[0] * buf[0, :R].numel()].view(buf.shape[0], R, *buf.shape[2:])


def exchange(eloc: torch.Tensor, comm, gat: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """E3: this rank's rows [R, layers, W] -> every rank's [R, layers, world * W] in out, rank 0's columns first.

    ``gat`` [world, rows, layers, W] receives the gather; world 1 returns ``eloc`` itself.
    """

    world, R = gat.shape[0], eloc.shape[0]
    if world == 1:
        return eloc
    recv = _lead(gat, R)
    fast_gather(comm, eloc, recv)
    full = out[:R]
    full.unflatten(-1, (world, -1)).copy_(recv.permute(1, 2, 0, 3))
    return full


def kv(e: torch.Tensor, wkv: Mx8Linear, comm, out: torch.Tensor, gat: torch.Tensor, *,
       prompt: bool = False) -> torch.Tensor:
    """L1: one layer's rows e [R, K] -> [keys | value] as world column blocks [world, R, (hc + 1) * D / world].

    Block r is rank r's rows of ``wkv`` (``out`` [rows, n] takes this rank's, ``gat`` [world, rows, n] the
    gather); world 1 is the whole projection, [1, R, (hc + 1) * D], with no gather.
    """

    R = e.shape[0]
    y = mx8.mm(wkv, e, out[:R], prompt=prompt)
    if gat.shape[0] == 1:
        return y[None]
    recv = _lead(gat, R)
    fast_gather(comm, y, recv)
    return recv


@triton.jit
def _gather_cols(KV, r, col, part_stride, row_stride, ok, P: tl.constexpr):
    """Columns ``col`` of row r of a projection held as column blocks of P."""

    return tl.load(KV + (col // P) * part_stride + r * row_stride + col % P, mask=ok, other=0.0).to(tl.float32)


@triton.jit
def _inject(X, KV, part_stride, row_stride, WQK, eps, clamp, scale, D: tl.constexpr, HC: tl.constexpr,
            P: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    ok = d < D
    xp = X + (r * HC + c) * D + d
    h = tl.load(xp, mask=ok, other=0.0).to(tl.float32)
    key = _gather_cols(KV, r, c * D + d, part_stride, row_stride, ok, P)
    value = _gather_cols(KV, r, HC * D + d, part_stride, row_stride, ok, P)
    w = tl.load(WQK + c * D + d, mask=ok, other=0.0)
    rh = tl.math.div_rn(1.0, tl.sqrt_rn(tl.math.div_rn(tl.sum(h * h, axis=0), D * 1.0) + eps))
    rk = tl.math.div_rn(1.0, tl.sqrt_rn(tl.math.div_rn(tl.sum(key * key, axis=0), D * 1.0) + eps))
    dot = tl.sum((h * w) * key, axis=0) * (rh * rk) * scale
    root = tl.sqrt_rn(tl.maximum(tl.abs(dot), clamp))
    # copysign, not sign: a dot of exactly zero still opens the gate by sqrt(clamp) on its sign bit's side (M:362)
    z = (root.to(tl.int32, bitcast=True) | (dot.to(tl.int32, bitcast=True) & -2147483648)).to(tl.float32,
                                                                                               bitcast=True)
    g = tl.math.div_rn(1.0, 1.0 + libdevice.exp(-z))
    tl.store(xp, (h + g * value).to(tl.bfloat16), mask=ok)


def inject(X: torch.Tensor, kv: torch.Tensor, wqk: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """N10b in place: X [R, hc, D] bf16 += gate * value per (row, copy), rounded to bf16 once (M:350-365).

    ``kv`` is ``kv()``'s [world, R, (hc + 1) * D / world] (a [R, (hc + 1) * D] tensor is one block); ``wqk`` is
    fp32 [hc, D], q_weight * k_weight.
    """

    if kv.dim() == 2:
        kv = kv[None]
    R, hc, D = X.shape
    world, rows, per = kv.shape
    if X.dtype != torch.bfloat16 or not X.is_contiguous() or kv.dtype != torch.bfloat16 or kv.stride(-1) != 1:
        raise ValueError("inject: X contiguous bf16 [R, hc, D], kv bf16 with unit column stride")
    if rows != R or world * per != (hc + 1) * D or wqk.dtype != torch.float32 or tuple(wqk.shape) != (hc, D):
        raise ValueError(f"inject: X {tuple(X.shape)}, kv {tuple(kv.shape)}, wqk {wqk.dtype} {tuple(wqk.shape)}")
    if R:
        block = triton.next_power_of_2(D)
        _inject[(R, hc)](X, kv, kv.stride(0), kv.stride(1), wqk.contiguous(), eps, CLAMP, D ** -0.5, D=D, HC=hc,
                         P=per, BLOCK=block, num_warps=16 if block > 4096 else 8, enable_fp_fusion=False)
    return X
