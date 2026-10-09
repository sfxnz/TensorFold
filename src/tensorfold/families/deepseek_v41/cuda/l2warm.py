"""Paced L2 warming of the dense weights read after each decode all-gather, on a side stream beside the gather."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import torch

WARM_ENV = "TF_DSV41_L2WARM"    # "RATE[:MB]": warm at RATE GB/s up to MB a gather (default CAP_MB); unset or 0: off
CAP_MB = 8
CTAS = 8                        # blocks of the warm kernel, together at the rate
SECTOR = 32                     # bytes one of its copies brings into L2
MAX_RANGES = 8                  # byte ranges one launch reads (``l2warm.cpp``)
_LOCAL = threading.local()


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_dsv41_l2warm_v1", sources=[str(here / "l2warm.cpp"), str(here / "l2warm.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def wanted(environ: Mapping[str, str] | None = None) -> tuple[float, int] | None:
    """(GB/s, bytes a gather) WARM_ENV asks for, None when off; ValueError naming the variable for a bad value."""

    value = (os.environ if environ is None else environ).get(WARM_ENV, "").strip()
    if value in ("", "0"):
        return None
    rate, colon, cap = value.partition(":")
    try:
        gbs, mb = float(rate), float(cap) if colon else float(CAP_MB)
    except ValueError:
        gbs = mb = math.nan
    if not (math.isfinite(gbs) and math.isfinite(mb) and gbs >= 1 and mb > 0):
        raise ValueError(f"{WARM_ENV}={value}: RATE[:MB], at least 1 GB/s and the MB a gather warms, e.g. 150 or 150:8")
    return gbs, int(mb * 2**20)


def _mx8(lin) -> list[torch.Tensor]:
    return [lin.bs, lin.w8]


def reads(w) -> list[tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """Per layer, the known weights read after each of its two gathers, in reading order: the MoE's after attention's
    share, the next block's (Engram, mHC, wq_a|wkv) or the head's after the MoE's."""

    out = []
    for i, lw in enumerate(w.layers):
        moe = [lw.hc_ffn.fn, lw.moe.gate, *_mx8(lw.moe.shared_gu)]
        if i + 1 < len(w.layers):
            nxt = w.layers[i + 1]
            eg = w.engram.get(nxt.index)
            after = [*(_mx8(eg.wkv) if eg is not None else []), nxt.hc_attn.fn, *_mx8(nxt.attn.wqa_kv)]
        else:
            after = [w.norm, *_mx8(w.head)]
        out.append((moe, after))
    return out


def ranges(tensors: Sequence[torch.Tensor], cap: int) -> list[tuple[int, int]]:
    """(address, sectors) of the whole 32-byte sectors inside the first ``cap`` bytes of ``tensors`` laid end to end."""

    out, left = [], cap
    for t in tensors:
        if left <= 0 or len(out) == MAX_RANGES:
            break
        a, take = t.data_ptr(), min(t.numel() * t.element_size(), left)
        lo, hi = -(-a // SECTOR) * SECTOR, (a + take) // SECTOR * SECTOR
        if hi > lo:
            out.append((lo, (hi - lo) // SECTOR))
        left -= take
    return out


def _side(device: torch.device) -> torch.cuda.Stream:
    """This thread's warm stream on ``device`` (two thread ranks on one GPU must not share it)."""

    streams = vars(_LOCAL).setdefault("streams", {})
    if device not in streams:
        streams[device] = torch.cuda.Stream(device)
    return streams[device]


class Warm:
    """Each gather's ranges as a device table, warmed on this thread's warm stream; a forward joins it at its end."""

    def __init__(self, w, gb_per_s: float, cap: int) -> None:
        self.rate, self.device = float(gb_per_s), torch.empty(0, device=w.device).device     # with its index
        self.tables: dict[tuple[int, int], torch.Tensor] = {}      # by (layer id, share 0 attention / 1 MoE)
        for lw, pair in zip(w.layers, reads(w)):
            for k, tensors in enumerate(pair):
                got = ranges(tensors, cap)
                if got:
                    self.tables[lw.index, k] = torch.tensor(got, dtype=torch.int64, device=self.device)
        _ext()

    def ahead(self, index: int, k: int) -> None:
        """Warm what layer ``index`` reads after its share ``k`` is gathered, from the current stream's point on."""

        table = self.tables.get((index, k))
        if table is None:
            return
        main, side = torch.cuda.current_stream(self.device), _side(self.device)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            _ext().warm(table, self.rate, CTAS)

    def join(self) -> None:
        """The current stream waits for every warm launched on this thread."""

        torch.cuda.current_stream(self.device).wait_stream(_side(self.device))
