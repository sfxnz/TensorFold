"""Engram on the device: file rows staged, dequantized, exchanged by column, projected and injected."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorfold.cuda.comm import fast_gather
from tensorfold.cuda.nvfp4.linear import Mx8Linear

from ..engram_table import WORKERS
from . import mx8
from .weights import EngramScales

CLAMP = 1e-6            # M:341: the gate's floor on |dot| before its square root


@lru_cache(maxsize=1)
def _io():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_dsv41_engram_io", sources=[str(here / "engram_io.cpp")], extra_cflags=["-O2"],
                verbose=False)


def read_pool(threads: int = WORKERS):
    """The native pread pool ``engram_table.Reader`` takes: ``threads`` threads, the caller's one of them."""

    return _io().Pool(threads)


def resident(ids: np.ndarray, scales: EngramScales) -> np.ndarray:
    """Layer-local row ids [..., layers, columns] -> int64 their rows in ``scales.rows``."""

    at = np.asarray(ids, dtype=np.int64) + np.asarray(scales.shift, dtype=np.int64)[:, None]
    if at.size and (at.min() < 0 or at.max() >= scales.rows.shape[0]):
        raise ValueError("Engram row ids outside this rank's resident hash columns")
    return at


def fill_rows(ids: np.ndarray, reader, rec: np.ndarray, scales: EngramScales | None = None,
              idx: np.ndarray | None = None) -> None:
    """Layer-local ids [n, layers, columns] -> file records u8 [n * layers * columns, W + S] in ``rec``; with
    ``scales`` the records' scale bytes are left as they are and each row's place in ``scales.rows`` goes to
    ``idx`` (int64 [n * layers * columns])."""

    starts = np.asarray(reader.layout.starts[:ids.shape[1]], dtype=np.int64)[:, None]
    flat = (ids.astype(np.int64) + starts).reshape(-1)
    if scales is None:
        reader.gather(flat, rec[:, :reader.wrow], rec[:, reader.wrow:])
    else:
        idx[:] = resident(ids, scales).reshape(-1)
        reader.gather(flat, rec[:, :reader.wrow])


def stage_rows(ids: np.ndarray, reader, host: torch.Tensor, done: torch.cuda.Event, eraw: torch.Tensor, *,
               scales: EngramScales | None = None, idx_host: torch.Tensor | None = None,
               idx: torch.Tensor | None = None, first: int = 0) -> int:
    """Rows ``ids`` [n, layers, columns] from the files into rows first..R = first + n of ``eraw`` through pinned half
    ``host`` -> R; with ``scales``, their places in ``scales.rows`` into ``idx`` through ``idx_host``. Rows before
    ``first`` are in the half already (``fill_rows`` after its copy's event ``done``)."""

    ids = np.asarray(ids)
    R = first + ids.shape[0]
    if ids.ndim != 3 or ids.shape[1:] != tuple(host.shape[1:3]) or R > min(host.shape[0], eraw.shape[0]):
        raise ValueError(f"Engram ids {ids.shape} from row {first} do not fit staging {tuple(host.shape)} -> "
                         f"{tuple(eraw.shape)}")
    per = ids.shape[1] * ids.shape[2]
    done.synchronize()
    if first < R:
        rec = host.numpy().reshape(-1, host.shape[-1])[first * per:R * per]
        fill_rows(ids, reader, rec, scales, idx_host.numpy().reshape(-1)[first * per:R * per] if scales is not None
                  else None)
    eraw[:R].copy_(host[:R], non_blocking=True)
    if scales is not None:
        idx[:R].copy_(idx_host[:R], non_blocking=True)
    done.record()
    return R


@triton.jit
def _dequant(RAW, OUT, SCALES, IDX, W: tl.constexpr, S: tl.constexpr, RESIDENT: tl.constexpr):
    """One file row: W e4m3 bytes then S E8M0 bytes (``RESIDENT``: row IDX[p] of SCALES) -> bf16(e4m3 * 2^(e - 127))
    per 32 (0xFF NaN, 0 the subnormal)."""

    p = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, W)
    v = tl.load(RAW + p * (W + S) + j).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    if RESIDENT:
        e = tl.load(SCALES + tl.load(IDX + p) * S + j // (W // S)).to(tl.int32)
    else:
        e = tl.load(RAW + p * (W + S) + W + j // (W // S)).to(tl.int32)
    s = tl.where(e == 0, 1 << 22, tl.where(e == 255, 0x7FC00000, e << 23)).to(tl.float32, bitcast=True)
    tl.store(OUT + p * W + j, (v * s).to(tl.bfloat16))


def dequant_rows(eraw: torch.Tensor, out: torch.Tensor, scales: EngramScales | None = None,
                 idx: torch.Tensor | None = None) -> torch.Tensor:
    """Dequant: u8 [R, layers, columns, W + W/32] -> out bf16 [R, layers, columns * W], exactly M:312-320's rows; with
    ``scales`` each row's scale bytes are row ``idx`` (int64 [R, layers, columns]) of ``scales.rows``."""

    R, layers, cols, width = eraw.shape
    w = width * 32 // 33
    if eraw.dtype != torch.uint8 or not eraw.is_contiguous() or w + w // 32 != width:
        raise ValueError(f"dequant_rows: eraw u8 [R, layers, columns, W + W/32], not {eraw.dtype} {tuple(eraw.shape)}")
    if out.dtype != torch.bfloat16 or not out.is_contiguous() or tuple(out.shape) != (R, layers, cols * w):
        raise ValueError(f"dequant_rows: out bf16 [{R}, {layers}, {cols * w}], not {out.dtype} {tuple(out.shape)}")
    if scales is not None and (idx is None or idx.dtype != torch.int64 or not idx.is_contiguous()
                               or tuple(idx.shape) != (R, layers, cols) or tuple(scales.rows.shape[1:]) != (w // 32,)):
        raise ValueError(f"dequant_rows: resident scales need idx int64 [{R}, {layers}, {cols}] into u8 [n, {w // 32}]")
    if R:
        resident = scales is not None
        _dequant[(R * layers * cols,)](eraw, out, scales.rows if resident else eraw, idx if resident else eraw, W=w,
                                       S=w // 32, RESIDENT=resident, num_warps=2)
    return out


def _lead(buf: torch.Tensor, R: int) -> torch.Tensor:
    """[world, R, ...] laid out contiguously at the start of ``buf``'s storage ([world, rows, ...], contiguous)."""

    return buf.view(-1)[:buf.shape[0] * buf[0, :R].numel()].view(buf.shape[0], R, *buf.shape[2:])


def exchange(eloc: torch.Tensor, comm, gat: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """This rank's rows [R, layers, W] -> every rank's [R, layers, world * W] in ``out``, rank 0's first."""

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
    """One layer's rows e [R, K] -> [keys | value] as world column blocks [world, R, (hc + 1) * D / world]."""

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
    """X [R, hc, D] bf16 += gate * value per (row, copy) in place, rounded to bf16 once (M:350-365)."""

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
