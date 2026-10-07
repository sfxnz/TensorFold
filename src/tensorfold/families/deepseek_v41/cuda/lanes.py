"""Several sequences' device state stacked lane-first, so each lane is an ordinary ``State`` over contiguous slices."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..config import Config
from . import MAX_ROWS
from .buffers import Buffers, State, _device_bytes

MAX_LANES = 4                   # sequences one shared verify forward serves


class Lanes:
    """``State``'s fields with a leading lane axis of ``slots``; ``view(k)`` is lane k's State, aliasing the stack."""

    def __init__(self, cfg: Config, slots: int, capacity: int, device: torch.device | str = "cuda") -> None:
        if not 0 < slots <= MAX_LANES:
            raise ValueError(f"lanes: {slots} slots, 1 to {MAX_LANES}")
        shape = State(cfg, capacity, "meta")         # one lane's fields, allocating nothing

        def stacked(t: torch.Tensor) -> torch.Tensor:
            return torch.zeros((slots, *t.shape), dtype=t.dtype, device=device)

        self.slots, self.capacity, self.device = slots, capacity, torch.device(device)
        self.ratio, self.pooled = shape.ratio, shape.pooled
        self.rings = stacked(shape.rings)
        self.comp = {layer: stacked(t) for layer, t in shape.comp.items()}
        self.index_k = {layer: stacked(t) for layer, t in shape.index_k.items()}
        self.tail, self.tail_valid = stacked(shape.tail), stacked(shape.tail_valid)
        self.pos_dev = torch.zeros((slots,), dtype=torch.int32, device=device)     # each lane's forward start
        self._views = tuple(State.view(self, k) for k in range(slots))

    def view(self, k: int) -> State:
        """Lane k as a State: its slices, its ``pos_dev`` element, its own host position and history."""

        return self._views[k]

    def nbytes(self) -> int:
        return _device_bytes(self, self.device)


class Ahead:
    """A lane's pinned Engram read-ahead block, decode's ``eraw_host[0]``, ``eidx_host[0]`` and ``eraw_done[0]``."""

    def __init__(self, cfg: Config, world: int) -> None:
        cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads // world
        raw = (MAX_ROWS, len(cfg.engram_layer_ids), cols, cfg.engram_head_dim + cfg.engram_head_dim // 32)
        self.eraw_host = [torch.zeros(raw, dtype=torch.uint8, pin_memory=True)]
        self.eidx_host = [torch.zeros(raw[:3], dtype=torch.int64, pin_memory=True)]
        self.eraw_done = [torch.cuda.Event()]


def check_room(b: Buffers, k: int, pos: int, rows: int) -> None:
    """Refuse lane k's window of ``rows`` rows at ``pos`` unless it fits a window and the lane's positions."""

    if not 0 < rows <= MAX_ROWS or not 0 <= pos <= b.capacity - rows:
        raise ValueError(f"lane {k}: {rows} rows at {pos}, 1 to {MAX_ROWS} in {b.capacity} positions")


def stage_tables(b: Buffers, windows: Sequence[tuple[int, int, Sequence[int]]]) -> int:
    """Ids and row tables of a shared forward over ``windows`` (lane, its position, its tokens) in ascending lane
    order, through the pinned twins once their previous copy has finished -> T, the rows."""

    lanes, T = [k for k, _, _ in windows], sum(len(tokens) for _, _, tokens in windows)
    if not windows or lanes != sorted(set(lanes)) or not 0 <= lanes[0] <= lanes[-1] < b.segs.shape[0] or T > b.rows:
        raise ValueError(f"lanes {lanes} with {T} rows: buffers of {b.segs.shape[0]} lanes and {b.rows} rows")
    for k, pos, tokens in windows:
        check_room(b, k, pos, len(tokens))
    b.staged.synchronize()
    ids, lane, rpos, seg0 = (getattr(b, f"{name}_host")[:T].numpy() for name in ("ids", "lane", "rpos", "seg0"))
    segs, r = b.segs_host.numpy(), 0
    for s, (k, pos, tokens) in enumerate(windows):
        R = len(tokens)
        ids[r:r + R], lane[r:r + R], rpos[r:r + R], seg0[r:r + R] = tokens, k, range(pos, pos + R), r
        segs[s] = r, R
        r += R
    b.nseg_host[0] = len(windows)
    for name in ("ids", "lane", "rpos", "seg0"):
        getattr(b, name)[:T].copy_(getattr(b, f"{name}_host")[:T], non_blocking=True)
    b.segs.copy_(b.segs_host, non_blocking=True)
    b.nseg.copy_(b.nseg_host, non_blocking=True)
    b.staged.record()
    return T
