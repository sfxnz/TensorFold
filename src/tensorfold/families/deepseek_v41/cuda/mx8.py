"""DeepSeek's FP8 projections on ``Mx8Linear``: bf16 outputs through it, fp32 outputs on its kernels' fp32 epilogue."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tensorfold.cuda.kernels import qmm
from tensorfold.cuda.nvfp4.linear import FUSED_ROWS, MXFP8, PROMPT_TILE, Mx8Linear, _ext, _prompt_ext

from . import gemv


def _folds(lin: Mx8Linear) -> bool:
    """``Mx8Linear.prefill``'s test: every exponent keeps an e4m3 byte times its power of two exact in bf16."""

    if lin.fold is None:
        lin.fold = bool(((lin.bs >= 3) & (lin.bs <= 134)).all())
    return lin.fold


def _f32(lin: Mx8Linear, x: torch.Tensor, out: torch.Tensor | None, prompt: bool) -> torch.Tensor:
    """The kernels ``lin`` / ``lin.prefill`` run, with their fp32 epilogue: the same sums, not rounded to bf16."""

    folded = prompt and _folds(lin)
    if x.dtype != torch.bfloat16 or x.stride(-1) != 1 or (folded and (x.data_ptr() % 16 or (
            x.shape[0] > 1 and x.stride(0) % 8))):
        x = x.to(torch.bfloat16).contiguous()
    m = x.shape[0]
    fits = out is not None and out.is_contiguous() and out.dtype == torch.float32 and tuple(out.shape) == (m, lin.n)
    y = out if fits else torch.empty((m, lin.n), dtype=torch.float32, device=x.device)
    if folded:
        _prompt_ext().prompt16(x, lin.w8, lin.bs, 1.0, y, MXFP8, lin.n, lin.npad, PROMPT_TILE, True)
    else:
        sk = qmm.split_k(lin.n, lin.k)
        part = torch.empty((sk, m, lin.n), dtype=torch.float32, device=x.device) if sk > 8 else None
        bm = 0 if sk > 1 and m >= FUSED_ROWS else qmm.bucket(m)
        _ext().qmmf(x, lin.w8, lin.bs, 1.0, y, part if bm else None, MXFP8, lin.n, sk, lin.npad, bm, True)
    if out is not None and y is not out:
        out.copy_(y)
    return y


def mm(lin: Mx8Linear, x: torch.Tensor, out: torch.Tensor | None = None, *, f32: bool = False,
       prompt: bool = False) -> torch.Tensor:
    """x (M, K) -> (M, n) bf16 as ``lin`` or (``prompt``) ``lin.prefill``, or fp32 before that rounding."""

    y = None if prompt else gemv.run(lin, x, out, f32)
    if y is not None:
        return y
    if f32:
        return _f32(lin, x, out, prompt)
    return lin.prefill(x, out) if prompt else lin(x, out)


class Groups(list):
    """``wo_a``'s group projections as views of ``whole``, their outputs stacked, which decode runs in one launch."""

    def __init__(self, whole: Mx8Linear, n: int) -> None:
        if whole.n % n or n % 128 or whole.npad != whole.n:
            raise ValueError(f"groups of {n} outputs do not split a [{whole.n}, {whole.k}] projection into 128s")
        tiles, per = n // 64, n * whole.k
        super().__init__(Mx8Linear(whole.w8[g * per:(g + 1) * per], whole.bs[g * tiles:(g + 1) * tiles], n, whole.k,
                                   n) for g in range(whole.n // n))
        self.whole = whole

    @classmethod
    def from_checkpoint(cls, weight: torch.Tensor, scale: torch.Tensor, n: int) -> Groups:
        """Groups of ``n`` rows of e4m3 ``weight`` [G n, K] with e8m0 ``scale`` bytes [G n, K/32]."""

        return cls(Mx8Linear.from_checkpoint(weight, scale), n)


def _stacked(groups: Groups, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """All groups in one decode launch: x's row r, group block g as row G r + g against every group's columns, each
    tile split over K as its group alone is, so the (r, g, g) blocks, copied to out, have each group's own bits."""

    G, k, n, whole = len(groups), groups[0].k, groups[0].n, groups.whole
    m = x.shape[0] * G
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        x = x.to(torch.bfloat16).contiguous()
    y = torch.empty((m, G * n), dtype=torch.bfloat16, device=x.device)
    sk = qmm.split_k(n, k)
    part = torch.empty((sk, m, G * n), dtype=torch.float32, device=x.device) if sk > 8 else None
    bm = 0 if sk > 1 and m >= FUSED_ROWS else qmm.bucket(m)
    _ext().qmmf(x.view(m, k), whole.w8, whole.bs, 1.0, y, part if bm else None, MXFP8, G * n, sk, whole.npad, bm,
                False)
    rows = x.shape[0]
    out.view(rows, G, n).copy_(y.as_strided((rows, G, n), (G * G * n, (G + 1) * n, 1)))
    return out


def grouped(lins: Sequence[Mx8Linear], x: torch.Tensor, out: torch.Tensor, *, prompt: bool = False) -> torch.Tensor:
    """``wo_a``: group g's projection of x's g-th block of columns into out's g-th block of columns, bf16."""

    k, n = lins[0].k, lins[0].n
    if x.shape[-1] != len(lins) * k or out.shape[-1] != len(lins) * n:
        raise ValueError(f"{len(lins)} groups of [{n}, {k}] do not tile x {tuple(x.shape)} -> out {tuple(out.shape)}")
    if isinstance(lins, Groups) and not prompt and out.is_contiguous():
        return _stacked(lins, x, out)
    for g, lin in enumerate(lins):
        mm(lin, x[:, g * k:(g + 1) * k], out[:, g * n:(g + 1) * n], prompt=prompt)
    return out
