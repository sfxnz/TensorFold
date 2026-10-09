"""DeepSeek's FP8 decode projections on qmmf's kernel with tiles picked by shape, each output's bits unchanged."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from tensorfold.cuda.kernels import qmm
from tensorfold.cuda.nvfp4.linear import Mx8Linear

KNOB = "TF_DSV41_QMMF"
COLUMNS, STAGES, REDUCE = (16, 32, 64), (4, 6, 8), ("cluster", "fuse")
MAX_ROWS = 32                       # one 16- or 32-row tile; more rows take Mx8Linear's own tiles


@dataclass(frozen=True)
class Tiles:
    """Columns a block, cp.async stages, and whether a tile's K slices meet in one block instead of a cluster."""

    bn: int = 64
    stages: int = 4
    fuse: bool = False


TUNED: dict[tuple[int, int] | None, Tiles] = {}      # (n, K) -> tiles when KNOB is unset; None: every other shape


def parse(spec: str) -> dict[tuple[int, int] | None, Tiles]:
    """``"NxK:bn/stages/cluster|fuse"`` entries split by commas, ``*`` for every shape not named; "" or "0": none."""

    table: dict[tuple[int, int] | None, Tiles] = {}
    if spec.strip() in ("", "0"):
        return table
    for entry in spec.split(","):
        try:
            shape, tiles = entry.strip().split(":")
            bn, stages, reduce = tiles.split("/")
            key = None if shape.strip() == "*" else tuple(int(v) for v in shape.lower().split("x"))
            t = Tiles(int(bn), int(stages), reduce.strip() == "fuse")
            ok = (key is None or len(key) == 2) and t.bn in COLUMNS and t.stages in STAGES and reduce.strip() in REDUCE
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(f"{KNOB}: {entry.strip()!r} is not NxK:bn/stages/reduce with bn in {COLUMNS}, stages in "
                             f"{STAGES} and reduce in {REDUCE} (or * for the shape)")
        table[key] = t
    return table


def tiles_wanted(environ: Mapping[str, str] | None = None) -> dict[tuple[int, int] | None, Tiles]:
    """KNOB's table, TUNED when it is unset."""

    value = (os.environ if environ is None else environ).get(KNOB)
    return dict(TUNED) if value is None else parse(value)


@lru_cache(maxsize=1)
def table() -> dict[tuple[int, int] | None, Tiles]:
    return tiles_wanted()


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here, nvfp4 = Path(__file__).parent, Path(__file__).parents[3] / "cuda" / "nvfp4"
    return load(name="tensorfold_dsv41_gemv_v2", sources=[str(here / "gemv.cpp"), str(here / "gemv.cu")],
                extra_include_paths=[str(nvfp4)], extra_cuda_cflags=["-O3"], verbose=False)


def run(lin: Mx8Linear, x: torch.Tensor, out: torch.Tensor | None, f32: bool) -> torch.Tensor | None:
    """x (M, K) through ``lin`` on its shape's tiles -> (M, n), fp32 or bf16; None for no tiles or over MAX_ROWS."""

    t = table()
    tiles = t.get((lin.n, lin.k), t.get(None))
    m = x.shape[0]
    if tiles is None or m > MAX_ROWS:
        return None
    if x.dtype != torch.bfloat16 or x.stride(-1) != 1 or x.data_ptr() % 16 or (m > 1 and x.stride(0) % 8):
        x = x.to(torch.bfloat16, memory_format=torch.contiguous_format, copy=True)
    dtype = torch.float32 if f32 else torch.bfloat16
    fits = out is not None and out.is_contiguous() and out.dtype == dtype and tuple(out.shape) == (m, lin.n)
    y = out if fits else torch.empty((m, lin.n), dtype=dtype, device=x.device)
    _ext().gemv(x, lin.w8, lin.bs, y, lin.n, qmm.split_k(lin.n, lin.k), lin.npad, tiles.bn, tiles.stages, tiles.fuse,
                f32)
    if out is not None and y is not out:
        out.copy_(y)
    return y
