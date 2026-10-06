"""DeepSeek-V4.1's activation quantize-dequantize per group of the last dim: bf16 values, or packed codes and scales.

A packed row holds the codes (e4m3 bytes, or e2m1 nibbles two to a byte, the first low), then one scale byte a group
(E8M0, or e4m3 for FP4 per 16). A packed value dequantizes to the bf16 value the QDQ gives, bit for bit: every e4m3
code or e2m1 code times its scale is exact in bf16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .norms import _rinv, _warps

GROUPS = 16                     # groups per program
INV448 = tl.constexpr(1 / 448)          # rounded to fp32 where it multiplies, as the reference's f32(1/448)
INV6 = tl.constexpr(1 / 6)
FLOOR8 = tl.constexpr(1e-4)             # amax floors (K:76, K:165, K:162)
FLOOR4_E8M0 = tl.constexpr(6 * 2.0**-126)
FLOOR4_E4M3 = tl.constexpr(6 * 2.0**-9)

FP8, FP4_E8M0, FP4_E4M3 = 0, 1, 2
GROUP = {FP8: 32, FP4_E8M0: 32, FP4_E4M3: 16}


@triton.jit
def _pow2_ceil(t):
    """K:22-33: 2^ceil(log2 t) from t's fp32 bits, the exponent plus one when the mantissa is non-zero."""

    bits = t.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _e2m1(v):
    """The nearest e2m1 value of |v| <= 6, ties to even (grid steps 0.5, 1, 2), keeping v's sign bit (-0 too)."""

    a = tl.abs(v)
    inv = tl.where(a < 2.0, 2.0, tl.where(a < 4.0, 1.0, 0.5))
    r = libdevice.rint(a * inv) * tl.where(a < 2.0, 0.5, tl.where(a < 4.0, 1.0, 2.0))
    sign = v.to(tl.int32, bitcast=True) & -2147483648
    return (r.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@triton.jit
def _quantized(x, KIND: tl.constexpr):
    """Each row of fp32 x [groups, G] quantized by the kind's rule: (grid values q, scales s [groups, 1]), fp32."""

    amax = tl.max(tl.abs(x), axis=1)
    if KIND == 0:                                   # K:70-86: FP8 e4m3, scale 2^ceil(log2(amax / 448))
        s = _pow2_ceil(tl.maximum(amax, FLOOR8) * INV448)[:, None]
        q = tl.minimum(tl.maximum(tl.math.div_rn(x, s), -448.0), 448.0)
        q = q.to(tl.float8e4nv).to(tl.float32)
    else:
        if KIND == 1:                               # K:165-166: FP4 e2m1, scale 2^ceil(log2(amax / 6))
            s = _pow2_ceil(tl.maximum(amax, FLOOR4_E8M0) * INV6)[:, None]
        else:                                       # K:162-163: FP4 e2m1, scale e4m3(amax / 6)
            s = tl.math.div_rn(tl.maximum(amax, FLOOR4_E4M3), 6.0).to(tl.float8e4nv).to(tl.float32)[:, None]
        q = _e2m1(tl.minimum(tl.maximum(tl.math.div_rn(x, s), -6.0), 6.0))
    return q, s


@triton.jit
def _dequantized(x, KIND: tl.constexpr):
    """Each row of fp32 x [groups, G] quantized and dequantized by the kind's rule, fp32."""

    q, s = _quantized(x, KIND)
    return q * s


@triton.jit
def _codes(q, KIND: tl.constexpr):
    """uint8 codes of grid values q [groups, G]: e4m3 bytes, or e2m1 nibbles (sign, then 0 0.5 1 1.5 2 3 4 6) paired."""

    if KIND == 0:
        out = q.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        a = tl.abs(q)
        m = tl.where(a < 2.0, a * 2.0, tl.where(a < 4.0, a + 2.0, a * 0.5 + 4.0)).to(tl.int32)
        c = m | ((q.to(tl.int32, bitcast=True) >> 28) & 8)
        lo, hi = tl.split(tl.reshape(c, [q.shape[0], q.shape[1] // 2, 2]))
        out = (lo | (hi << 4)).to(tl.uint8)
    return out


@triton.jit
def _scale_bytes(s, KIND: tl.constexpr):
    """One byte per scale of s [groups]: e4m3 for FP4 per 16, else the E8M0 exponent."""

    if KIND == 2:
        return s.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    return ((s.to(tl.int32, bitcast=True) >> 23) & 0xFF).to(tl.uint8)


@triton.jit
def _e2m1_value(c):
    """fp32 value of int32 e2m1 codes, -0 for code 8."""

    m = c & 7
    f = m.to(tl.float32)
    a = tl.where(m < 4, f * 0.5, tl.where(m < 6, f - 2.0, (f - 4.0) * 2.0))
    return (a.to(tl.int32, bitcast=True) | ((c & 8) << 28)).to(tl.float32, bitcast=True)


@triton.jit
def as_loaded(x):
    """bf16 x through an opaque move: a dot then lays it out as a bf16 load, never by the bytes it was unpacked from
    (which reorders K within the MMA steps and so the fp32 sums); a compiler choice, so the packed == bf16 tests
    guard it."""

    return tl.inline_asm_elementwise("mov.b16 $0, $1;", "=h,h", [x], dtype=tl.bfloat16, is_pure=False, pack=1)


@triton.jit
def unpack_fp8(ROW, ok, N: tl.constexpr, D: tl.constexpr):
    """fp32 [N, D] of the packed FP8 rows starting at pointers ``ROW`` [N], zeros where not ``ok``; each element
    reads its own scale byte, so no value moves between threads."""

    k = tl.arange(0, D)[None, :]
    c = tl.load(ROW[:, None] + k, mask=ok[:, None], other=0)
    e = tl.load(ROW[:, None] + D + k // 32, mask=ok[:, None], other=0)
    return c.to(tl.float8e4nv, bitcast=True).to(tl.float32) * (e.to(tl.int32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def unpack_fp4(ROW, ok, N: tl.constexpr, D: tl.constexpr, G: tl.constexpr):
    """fp32 [N, D] of the packed FP4 rows starting at pointers ``ROW`` [N] (per 16: e4m3 scales, per 32: E8M0); each
    element reads its own code byte and scale byte."""

    k = tl.arange(0, D)[None, :]
    b = tl.load(ROW[:, None] + k // 2, mask=ok[:, None], other=0).to(tl.int32)
    e = tl.load(ROW[:, None] + D // 2 + k // G, mask=ok[:, None], other=0)
    if G == 16:
        s = e.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        s = (e.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    return _e2m1_value((b >> ((k & 1) * 4)) & 15) * s


@triton.jit
def _qdq(X, Y, n, per_row, sx, sy, G: tl.constexpr, KIND: tl.constexpr, B: tl.constexpr, CB: tl.constexpr,
         PACKED: tl.constexpr):
    g = tl.program_id(0) * B + tl.arange(0, B)
    ok = (g < n)[:, None]
    row, col = (g // per_row).to(tl.int64), (g % per_row) * G
    j = tl.arange(0, G)[None, :]
    x = tl.load(X + row[:, None] * sx + col[:, None] + j, mask=ok, other=0.0).to(tl.float32)
    if PACKED:                                      # group i's CB code bytes at i * CB, its scale byte after the codes
        q, s = _quantized(x, KIND)
        at = Y + row * sy
        tl.store(at[:, None] + (g % per_row)[:, None] * CB + tl.arange(0, CB)[None, :], _codes(q, KIND), mask=ok)
        tl.store(at + per_row * CB + g % per_row, _scale_bytes(tl.reshape(s, [B]), KIND), mask=g < n)
    else:
        tl.store(Y + row[:, None] * sy + col[:, None] + j, _dequantized(x, KIND).to(tl.bfloat16), mask=ok)


@triton.jit
def _neg(v):
    """-v by its sign bit (Triton's unary minus is 0 - v, which keeps +0 for +0)."""

    return (v.to(tl.int32, bitcast=True) ^ -2147483648).to(tl.float32, bitcast=True)


@triton.jit
def _norm_rope_fp8(X, sx, W, OUT, so, P, T, eps, D: tl.constexpr, HALF: tl.constexpr, PER_ROW: tl.constexpr,
                   PACKED: tl.constexpr):
    """One row: rmsnorm's, rope.apply's and fp8_qdq_1x32's arithmetic and bf16 roundings in one program."""

    row = tl.program_id(0)
    r = row.to(tl.int64)
    d = tl.arange(0, D)
    ok = d < D
    rinv = _rinv(tl.load(X + r * sx + d, mask=ok, other=0.0).to(tl.float32), eps, D)
    e = tl.arange(0, D // 32)[:, None] * 32 + tl.arange(0, 32)[None, :]      # QDQ groups of 32, as rows
    pe = e ^ 1                                                                  # each element's RoPE partner
    y = (tl.load(W + e).to(tl.float32) * (tl.load(X + r * sx + e).to(tl.float32) * rinv)).to(tl.bfloat16)
    yp = (tl.load(W + pe).to(tl.float32) * (tl.load(X + r * sx + pe).to(tl.float32) * rinv)).to(tl.bfloat16)
    y, yp = y.to(tl.float32), yp.to(tl.float32)
    if PER_ROW:
        p = tl.load(P + row).to(tl.int64)
    else:
        p = tl.load(P).to(tl.int64) + row
    rot = e >= D - 2 * HALF                                                     # the last 2 * HALF dims
    i = tl.maximum(e - (D - 2 * HALF), 0) // 2
    c = tl.load(T + p * (2 * HALF) + 2 * i, mask=rot, other=0.0)
    s = tl.load(T + p * (2 * HALF) + 2 * i + 1, mask=rot, other=0.0)
    even = (e & 1) == 0
    a = tl.where(even, y, yp)
    b = tl.where(even, yp, y)
    # the pair (a, b) -> (a c - b s, a s + b c) with the products b s and b c rounded, as rope._rope compiles
    v = tl.where(even, tl.fma(a, c, _neg(b * s)), tl.fma(a, s, b * c))
    v = tl.where(rot, v.to(tl.bfloat16).to(tl.float32), y)
    if PACKED:
        q, s = _quantized(v, 0)
        tl.store(OUT + r * so + e, _codes(q, 0))
        tl.store(OUT + r * so + D + tl.arange(0, D // 32), _scale_bytes(tl.reshape(s, [D // 32]), 0))
    else:
        tl.store(OUT + r * so + e, _dequantized(v, 0).to(tl.bfloat16))


def width(d: int, kind: int) -> int:
    """Bytes of one packed row of ``d`` values: the codes, then a scale byte a group."""

    return (d if kind == FP8 else d // 2) + d // GROUP[kind]


def _packed(out: torch.Tensor, shape: tuple[int, ...], kind: int) -> bool:
    """Whether ``out`` takes packed rows of bf16 ``shape``, which it must match one way or the other."""

    if out.dtype == torch.uint8 and tuple(out.shape) == (*shape[:-1], width(shape[-1], kind)):
        return True
    if out.dtype == torch.bfloat16 and tuple(out.shape) == tuple(shape):
        return False
    raise ValueError(f"quant: out {out.dtype} {tuple(out.shape)} is neither bf16 {tuple(shape)} nor its packed rows")


def unpack(rows: torch.Tensor, kind: int, d: int) -> torch.Tensor:
    """bf16 [..., d] of packed uint8 rows (torch, any device): the QDQ's values, bit for bit."""

    group = GROUP[kind]
    if rows.dtype != torch.uint8 or rows.shape[-1] != width(d, kind):
        raise ValueError(f"unpack: uint8 rows of {width(d, kind)} bytes, not {rows.dtype} {tuple(rows.shape)}")
    codes, e = rows[..., :width(d, kind) - d // group], rows[..., width(d, kind) - d // group:]
    if kind == FP8:
        v = codes.contiguous().view(torch.float8_e4m3fn).float()
    else:
        c = torch.stack([codes & 15, codes >> 4], -1).flatten(-2).long()
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=rows.device)[c & 7]
        v = torch.where(c >= 8, -grid, grid)
    s = e.contiguous().view(torch.float8_e4m3fn).float() if kind == FP4_E4M3 else \
        (e.int() << 23).view(torch.float32)
    return (v.unflatten(-1, (-1, group)) * s[..., None]).flatten(-2).to(torch.bfloat16)


def _run(x: torch.Tensor, out: torch.Tensor | None, group: int, kind: int) -> torch.Tensor:
    out = x if out is None else out
    if x.dtype != torch.bfloat16:
        raise ValueError("quant: bf16 input")
    d = x.shape[-1]
    if d % group:
        raise ValueError(f"quant: the last dim {d} is not a multiple of the group {group}")
    packed = _packed(out, tuple(x.shape), kind)
    xs, ys = x.view(-1, d), out.view(-1, out.shape[-1])
    if xs.stride(-1) != 1 or ys.stride(-1) != 1:
        raise ValueError("quant: the last dim needs unit stride")
    n = xs.numel() // group
    if n:
        _qdq[(triton.cdiv(n, GROUPS),)](xs, ys, n, d // group, xs.stride(0), ys.stride(0), G=group, KIND=kind,
                                        B=GROUPS, CB=group if kind == FP8 else group // 2, PACKED=packed)
    return out


def fp8_qdq_1x32(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """FP8 e4m3 per 32 with a power-of-two scale (the window KV); in place unless ``out`` (bf16, or uint8 packed)."""

    return _run(x, out, 32, FP8)


def fp4_qdq_1x32_e8m0(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """FP4 e2m1 per 32 with a power-of-two scale (index Q and index-K); in place unless ``out`` (bf16, or packed)."""

    return _run(x, out, 32, FP4_E8M0)


def fp4_qdq_1x16_e4m3(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """FP4 e2m1 per 16 with an e4m3 scale (compressed KV entries); in place unless ``out`` (bf16, or packed)."""

    return _run(x, out, 16, FP4_E4M3)


def norm_rope_fp8(x: torch.Tensor, w: torch.Tensor, eps: float, positions: torch.Tensor, table: torch.Tensor,
                  out: torch.Tensor) -> torch.Tensor:
    """The window KV write in one launch: x [R, D] -> out [R, D] bf16 (or its packed rows), bit for bit
    ``fp8_qdq_1x32(rope.apply(norms.rmsnorm(x, w, eps, out), positions, table))``."""

    rows, d = x.shape
    half = table.shape[1]
    if x.dtype != torch.bfloat16 or d & (d - 1) or d < 32:
        raise ValueError(f"norm_rope_fp8: bf16 x [R, D], D a power of two >= 32, not {x.dtype} {tuple(x.shape)}")
    packed = _packed(out, (rows, d), FP8)
    if x.stride(-1) != 1 or out.stride(-1) != 1 or table.dtype != torch.float32 or not table.is_contiguous() \
            or 2 * half > d:
        raise ValueError("norm_rope_fp8: unit-stride rows and a contiguous fp32 [capacity, rope/2, 2] table")
    if positions.dim() != 1 or positions.numel() not in (1, rows) or positions.is_floating_point():
        raise ValueError(f"norm_rope_fp8: integer positions [1] or [{rows}], got {tuple(positions.shape)}")
    if rows:
        _norm_rope_fp8[(rows,)](x, x.stride(0), w, out, out.stride(0), positions, table, eps, D=d, HALF=half,
                                PER_ROW=positions.numel() == rows, PACKED=packed, num_warps=_warps(d),
                                enable_fp_fusion=False)
    return out
