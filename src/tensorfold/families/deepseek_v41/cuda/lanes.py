"""Several sequences' device state stacked lane-first, so each lane is an ordinary ``State`` over contiguous slices."""

from __future__ import annotations

import torch

from ..config import Config
from .buffers import State, _device_bytes

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
