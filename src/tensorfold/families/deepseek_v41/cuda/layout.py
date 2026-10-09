"""A debug dump of where a lane decoder's device buffers and the allocator's segments sit, to compare two builds."""

from __future__ import annotations

import json
import os

import torch

ENV = "TF_DSV41_LAYOUT"     # a path prefix: each dump writes <prefix>.<tag>.rank<r>.json


def tensors(groups: dict[str, object]) -> dict[str, list[int]]:
    """Every device tensor among each group's attributes (dicts, lists and tuples of them too) -> name: [address,
    bytes]."""

    out: dict[str, list[int]] = {}

    def add(name: str, v) -> None:
        if isinstance(v, torch.Tensor):
            if v.device.type != "cpu":
                out[name] = [v.data_ptr(), v.numel() * v.element_size()]
        elif isinstance(v, dict):
            for key, x in v.items():
                add(f"{name}[{key}]", x)
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                add(f"{name}[{i}]", x)

    for group, obj in groups.items():
        for attr, v in vars(obj).items() if obj is not None else ():
            add(f"{group}.{attr}", v)
    return out


def groups(dec) -> dict[str, object]:
    """The decoder's shared buffers, lanes, each lane's drafter scratch, the kept arena and the batch scratch."""

    found = {"mbuf": dec.mbuf, "pbuf": dec.pbuf, "lanes": dec.lanes, "kept": dec.kept, "batch": dec.batch,
             "warm": getattr(dec.w, "warm", None)}
    found.update({f"dwork{k}": e.dwork for k, e in enumerate(dec.engines)})
    return found


def segments() -> list[dict]:
    """The caching allocator's segments by address: size, bytes in use, type and pool."""

    keys = ("address", "total_size", "allocated_size", "segment_type", "segment_pool_id")
    return sorted(({k: s.get(k) for k in keys} for s in torch.cuda.memory_snapshot()), key=lambda s: s["address"])


def dump(tag: str, dec) -> str | None:
    """With ENV set, ``dec``'s buffers and the segments into <prefix>.<tag>.rank<r>.json -> that path."""

    prefix = os.environ.get(ENV)
    if not prefix:
        return None
    path = f"{prefix}.{tag}.rank{dec.w.rank}.json"
    with open(path, "w") as f:
        json.dump({"tensors": tensors(groups(dec)), "segments": segments()}, f, indent=1)
    return path
