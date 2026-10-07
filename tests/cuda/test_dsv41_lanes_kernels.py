"""LANES attention and compressor on stacked lanes: each row equals its lane's solo call on that lane's slices, bit for
bit, over random segmentations and positions, on the tiny geometry and the pack's sharded over two ranks."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_tiny

from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import attn_kernel as ak
from tensorfold.families.deepseek_v41.cuda import buffers, compressor, quant, rope
from tensorfold.families.deepseek_v41.cuda.lanes import Lanes
from tensorfold.families.deepseek_v41.cuda.weights import CompW, IdxW
from tensorfold.families.glm5_next.cuda.qmm import make_b16

GEOMETRY = {"tiny": (Config.from_dict(dsv41_tiny.config()), 1),
            "pack": (Config.read(Path(__file__).parents[1] / "fixtures" / "deepseek_v41"), 2)}
CAPACITY = 8192
SPOTS = (0, 1, 2, 127, 128, 5000)
# (lane, forward start, rows) of the live lanes, ascending: a lane at 0 beside one at 5000, and the ring's edge
EDGES = [[(0, 0, 3), (1, 5000, 6)], [(0, 5000, 1), (2, 0, 6)], [(0, 127, 6), (1, 128, 5), (2, 0, 1), (3, 5000, 2)],
         [(1, 1, 4), (3, 4999, 6)]]


def _plan(slots: int, seed: int) -> list[tuple[int, int, int]]:
    """A random segmentation: a non-empty set of live lanes, each at a spot or a random odd or even position."""

    g = random.Random(seed)
    live = sorted(g.sample(range(slots), g.randint(1, slots)))
    return [(k, g.choice([*SPOTS, 2 * g.randrange(3000), 2 * g.randrange(3000) + 1]), g.randint(1, 6)) for k in live]


PLANS = [pytest.param(4, p, id=f"edge{i}") for i, p in enumerate(EDGES)] + [
    pytest.param(s, _plan(s, 10 * s + i), id=f"S{s}-{i}") for s in (2, 3, 4) for i in range(3)]


def _stage(lanes: Lanes, buf: buffers.Buffers, plan: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Each live lane at its start and the row tables of its segment; returns (lane, seg0, rows)."""

    lane, rpos, seg0, out = [], [], [], []
    for k, p, rows in plan:
        lanes.view(k).set_pos(p)
        out.append((k, len(lane), rows))
        seg0 += [len(lane)] * rows
        lane += [k] * rows
        rpos += range(p, p + rows)
    for name, v in (("lane", lane), ("rpos", rpos), ("seg0", seg0)):
        getattr(buf, name)[:len(v)].copy_(torch.tensor(v, dtype=torch.int32))
    return out


def _packed(n: int, d: int, kind: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn((n, d), generator=gen).to(torch.bfloat16).cuda()
    return quant._run(x, torch.empty((n, quant.width(d, kind)), dtype=torch.uint8, device="cuda"),
                      quant.GROUP[kind], kind)


def _same(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), what


def _layers(cfg: Config) -> list[int]:
    """A window layer, then the first ratio-2 and the first ratio-1 compressed layers."""

    ratios = [cfg.roles[i].ratio for i in range(cfg.num_hidden_layers)]
    return [ratios.index(0), ratios.index(2), ratios.index(1)]


@pytest.mark.parametrize("S,plan", PLANS)
@pytest.mark.parametrize("geometry", GEOMETRY)
def test_attention_rows_equal_their_lane_alone(geometry, S, plan):
    cfg, world = GEOMETRY[geometry]
    gen, D = torch.Generator().manual_seed(len(plan)), cfg.head_dim
    H, W, WIN = cfg.num_attention_heads // world, cfg.index_topk, cfg.sliding_window
    lanes, buf = Lanes(cfg, S, CAPACITY), buffers.Buffers(cfg, world, 6 * S, CAPACITY, lanes=S)
    segs = _stage(lanes, buf, plan)
    T = sum(rows for _, _, rows in plan)
    rpos = buf.rpos[:T].tolist()
    for L in _layers(cfg):
        ratio, src = cfg.roles[L].ratio, cfg.roles[L].kv_src
        lanes.rings[:, L] = _packed(S * WIN, D, quant.FP8, gen).view(S, WIN, -1)
        buf.kvw[L, :T] = _packed(T, D, quant.FP8, gen)
        q = (2 * torch.randn((T, H, D), generator=gen)).to(torch.bfloat16).cuda()
        sink = torch.randn(H, generator=gen).cuda()
        comp = lists = counts = None
        if ratio:
            comp = lanes.comp[src]
            comp.copy_(_packed(S * comp.shape[1], D, quant.FP4_E4M3, gen).view(comp.shape))
            lists = torch.full((T, W), -1, dtype=torch.int32)
            counts = torch.zeros(T, dtype=torch.int32)
            for r, p in enumerate(rpos):
                pick = torch.randperm((p + 1) // ratio, generator=gen)[:W].sort().values
                lists[r, :len(pick)], counts[r] = pick.int(), len(pick)
            lists, counts = lists.cuda(), counts.cuda()
        nch = ak.chunks(W if ratio else 0, WIN)
        out = torch.empty_like(q)
        args = ak.LaneArgs(buf.lane, buf.seg0, lanes.pos_dev, lanes.rings.stride(0), comp.stride(0) if ratio else 0)
        ak.attention(q, lanes.rings[0, L], buf.kvw[L], lanes.pos_dev[:1], buf.rpos, comp[0] if ratio else None, lists,
                     counts, sink, out, prompt=False, part=torch.empty((T, H, nch, D + 2), device="cuda"), lanes=args)
        for k, s0, R in segs:
            rows = slice(s0, s0 + R)
            alone = torch.empty((R, H, D), dtype=torch.bfloat16, device="cuda")
            ak.attention(q[rows], lanes.rings[k, L], buf.kvw[L, rows], lanes.pos_dev[k:k + 1], buf.rpos[rows],
                         comp[k] if ratio else None, lists[rows] if ratio else None, counts[rows] if ratio else None,
                         sink, alone, prompt=False, part=torch.empty((R, H, nch, D + 2), device="cuda"))
            _same(out[rows], alone, f"layer {L}, lane {k} at {rpos[s0]}, {R} rows")


def _compressors(cfg: Config, gen: torch.Generator) -> list[SimpleNamespace]:
    """Random bf16 compressor and index-K weights of every KV source, as ``compress`` reads them."""

    D, hd, dk = cfg.hidden_size, cfg.head_dim, cfg.index_head_dim
    out = []
    for layer in cfg.kv_source_layer_ids:
        ratio = cfg.roles[layer].ratio
        wkv = torch.randn((ratio * hd if ratio == 2 else hd, D), generator=gen) * D**-0.5
        comp = CompW(ratio, make_b16(wkv.cuda()), (1 + 0.5 * torch.randn(hd, generator=gen)).bfloat16().cuda())
        wk = make_b16((torch.randn((dk, hd), generator=gen) * hd**-0.5).cuda())
        idx = IdxW(None, None, wk, (1 + 0.5 * torch.randn(dk, generator=gen)).bfloat16().cuda())
        out.append(SimpleNamespace(index=layer, attn=SimpleNamespace(comp=comp, idx=idx)))
    return out


def _fields(lanes: Lanes) -> list[torch.Tensor]:
    return [*lanes.comp.values(), *lanes.index_k.values(), lanes.tail, lanes.tail_valid]


@pytest.mark.parametrize("S,plan", PLANS)
@pytest.mark.parametrize("geometry", GEOMETRY)
def test_compressor_rows_equal_their_lane_alone(geometry, S, plan):
    cfg, world = GEOMETRY[geometry]
    gen = torch.Generator().manual_seed(7 + len(plan))
    lanes, solo = Lanes(cfg, S, CAPACITY), Lanes(cfg, S, CAPACITY)
    for t in _fields(lanes):       # cache bytes, tails and their flags: any values, so a stray write shows
        t.copy_(torch.randint(0, 256, t.shape, generator=gen, dtype=torch.uint8) if t.dtype == torch.uint8 else
                torch.randn(t.shape, generator=gen).to(t.dtype))
    for a, b in zip(_fields(solo), _fields(lanes)):
        a.copy_(b)
    buf, one = buffers.Buffers(cfg, world, 6 * S, CAPACITY, lanes=S), buffers.Buffers(cfg, world, 6, CAPACITY)
    segs = _stage(lanes, buf, plan)
    T = sum(rows for _, _, rows in plan)
    xa = torch.randn((T, cfg.hidden_size), generator=gen).to(torch.bfloat16).cuda()
    table, eps = rope.tables(cfg, "yarn", CAPACITY, "cuda"), cfg.rms_norm_eps
    for lw in _compressors(cfg, gen):
        compressor.compress(lw, xa, lanes, buf, table, eps, tables=(buf.lane[:T], buf.rpos[:T], buf.seg0[:T]))
        for k, s0, R in segs:
            st = solo.view(k)
            st.set_pos(int(lanes.pos_dev[k]))
            compressor.compress(lw, xa[s0:s0 + R], st, one, table, eps)
            what = f"layer {lw.index}, lane {k} at {st.pos}, {R} rows"
            _same(buf.lat[s0:s0 + R], one.lat[:R], what + " latents")
            _same(buf.kI[s0:s0 + R], one.kI[:R], what + " index keys")
        for i, (a, b) in enumerate(zip(_fields(lanes), _fields(solo))):
            _same(a, b, f"layer {lw.index}: cache {i}")
