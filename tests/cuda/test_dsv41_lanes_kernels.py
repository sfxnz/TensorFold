"""LANES attention, compressor and indexer on stacked lanes: each row equals its lane's solo call on that lane's slices,
bit for bit, over random segmentations and positions, on the tiny geometry and the pack's sharded over two ranks."""

from __future__ import annotations

import random
from functools import partial
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
from tensorfold.families.deepseek_v41.cuda import buffers, compressor, indexer, quant, rope
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


def _layout(plan: list[tuple[int, int, int]]) -> tuple[list[int], list[int], list[int], list[tuple[int, int, int]]]:
    """The row tables lane, rpos and seg0 of a plan, and its (lane, seg0, rows) segments."""

    lane, rpos, seg0, out = [], [], [], []
    for k, p, rows in plan:
        out.append((k, len(lane), rows))
        seg0 += [len(lane)] * rows
        lane += [k] * rows
        rpos += range(p, p + rows)
    return lane, rpos, seg0, out


def _stage(lanes: Lanes, buf: buffers.Buffers, plan: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Each live lane at its start and the row tables of its segment; returns (lane, seg0, rows)."""

    for k, p, _ in plan:
        lanes.view(k).set_pos(p)
    lane, rpos, seg0, out = _layout(plan)
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


IDX_CAPACITY = 36864
# (lane, about the entries its first row sees, rows): a lane of 16 beside one of about 2500, and one past the pack's
# candidate blocks (2048 of 8 entries)
IDX_EDGES = [[(0, 16, 6), (1, 2500, 6)], [(1, 2500, 1), (3, 15, 4)], [(0, 0, 2), (1, 17000, 6), (2, 16, 3), (3, 2499, 5)]]
IDX_PLANS = [pytest.param(4, p, id=f"edge{i}") for i, p in enumerate(IDX_EDGES)] + [
    pytest.param(s, _plan(s, 100 + 10 * s + i), id=f"S{s}-{i}") for s in (2, 3, 4) for i in range(3)]


def _i32(v) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.int32, device="cuda")


def _lane_tables(plan: list[tuple[int, int, int]], S: int, keys: torch.Tensor):
    """The indexer's tables of a plan, the segments past its count poisoned; returns them and the segments."""

    lane, rpos, _, segs = _layout(plan)
    rows = [(s0, R) for _, s0, R in segs] + [(-7, -7)] * (S - len(segs))
    return indexer.LaneArgs(_i32(lane), _i32(rpos), _i32(rows), _i32([len(segs)]), keys.stride(0)), segs


def _select(cfg: Config, mode: str, q, w, keys, ratio: int, pos, within, work, tables=None) -> SimpleNamespace:
    """One ``indexer.select`` into outputs poisoned so an unwritten value shows."""

    R, BLK = q.shape[0], cfg.candidate_block_size
    width = max(keys.shape[0], cfg.candidate_topk_blocks * BLK)
    o = SimpleNamespace(s=torch.full((R, width), float("nan"), device="cuda"),
                        lists=torch.full((R, cfg.index_topk), -7, dtype=torch.int32, device="cuda"),
                        cand=torch.full((R, cfg.candidate_topk_blocks), -7, dtype=torch.int32, device="cuda"),
                        n=torch.full((R,), -7, dtype=torch.int32, device="cuda"))
    o.cand_n = o.n.clone()
    indexer.select(cfg, q, w, keys, ratio, pos, o.s, o.lists, o.n, source=(o.cand, o.cand_n) if mode == "source"
                   else None, within=within, work=work, tables=tables)
    return o


def _index_inputs(cfg: Config, T: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """FP4-grid bf16 index queries and fp32 weights of T rows."""

    q = torch.randn((T, cfg.index_n_heads, cfg.index_head_dim), generator=gen).to(torch.bfloat16).cuda()
    return quant.fp4_qdq_1x32_e8m0(q), (torch.randn((T, cfg.index_n_heads), generator=gen) * 0.05).cuda()


@pytest.mark.parametrize("split", [False, True], ids=["rows", "split"])
@pytest.mark.parametrize("mode", ["full", "source", "within"])
@pytest.mark.parametrize("S,plan", IDX_PLANS)
@pytest.mark.parametrize("geometry", GEOMETRY)
def test_indexer_rows_equal_their_lane_alone(geometry, S, plan, mode, split):
    """Scores over each row's visible entries, candidate blocks and top-k lists with their counts: a ratio-2 index
    layer, the ratio-1 candidate source, and a reindex layer over the source's blocks."""

    cfg = GEOMETRY[geometry][0]
    ratio, BLK = 2 if mode == "full" else 1, cfg.candidate_block_size
    gen = torch.Generator().manual_seed(31 * S + len(plan))
    plan = [(k, e * ratio + k % ratio, rows) for k, e, rows in plan]
    E, slots = buffers.entries(ratio, IDX_CAPACITY), cfg.candidate_topk_blocks * BLK
    keys = _packed(S * E, cfg.index_head_dim, quant.FP4_E8M0, gen).view(S, E, -1)
    tables, segs = _lane_tables(plan, S, keys)
    T, rpos = tables.lane.shape[0], tables.rpos.tolist()
    work = (buffers.split_work(T, E, slots, "cuda"), buffers.split_work(6, E, slots, "cuda")) if split else (None,) * 2
    within = None
    if mode == "within":                              # the source's blocks for the stacked rows
        src = _select(cfg, "source", *_index_inputs(cfg, T, gen), keys[0], 1, None, None, None, tables)
        within = (src.cand, src.cand_n)
    q, w = _index_inputs(cfg, T, gen)
    out = _select(cfg, mode, q, w, keys[0], ratio, None, within, work[0], tables)
    for k, s0, R in segs:
        rows = slice(s0, s0 + R)
        mine = (within[0][rows], within[1][rows]) if within else None
        alone = _select(cfg, mode, q[rows], w[rows], keys[k], ratio, _i32([rpos[s0]]), mine, work[1])
        for r in range(R):
            vis, what = (rpos[s0 + r] + 1) // ratio, f"{mode}: lane {k} row {r} at {rpos[s0 + r]}"
            if within:                                # slot c holds entry cand[c // BLK] * BLK + c % BLK
                c = torch.arange(int(within[1][s0 + r]) * BLK, device="cuda")
                seen = c[within[0][s0 + r, c // BLK] * BLK + c % BLK < vis]
                _same(out.s[s0 + r, seen], alone.s[r, seen], what + " scores")
            else:
                _same(out.s[s0 + r, :vis], alone.s[r, :vis], what + " scores")
            n = int(alone.n[r])
            assert int(out.n[s0 + r]) == n and torch.equal(out.lists[s0 + r, :n], alone.lists[r, :n]), what
            if mode == "source":
                n = int(alone.cand_n[r])
                assert int(out.cand_n[s0 + r]) == n and torch.equal(out.cand[s0 + r, :n], alone.cand[r, :n]), what


def _selection(mode: str, s: torch.Tensor, lists, n, cand, cand_n, block: int, work, tables) -> None:
    """A decode forward's selection on its scores: blocks then lists (source), lists of the blocks (within), lists."""

    ratio, pos = 2 if mode == "full" else 1, tables.rpos
    if mode == "source":
        indexer.candidates(s, ratio, pos, cand, cand_n, block=block, work=work, tables=tables)
    within = (cand, cand_n) if mode == "within" else (None, None)
    indexer.topk(s, ratio, pos, lists, n, cand=within[0], cand_n=within[1], block=block, work=work, tables=tables)


def _median_ms(step, reps: int = 10) -> float:
    step()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        step()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[reps // 2]


def test_selection_time_of_the_largest_lane_forward():
    """mp7, printed: the LANES selection of four lanes of six rows at the end of 1,048,576 slots, the pack's geometry."""

    cfg, C, gen = GEOMETRY["pack"][0], 1 << 20, torch.Generator().manual_seed(5)
    BLK, BLOCKS, K = cfg.candidate_block_size, cfg.candidate_topk_blocks, cfg.index_topk
    for mode, ratio in (("full", 2), ("source", 1), ("within", 1)):
        E = buffers.entries(ratio, C)
        tables, _ = _lane_tables([(k, C - 6, 6) for k in range(4)], 4, torch.empty((4, E, 1), dtype=torch.uint8))
        T = tables.lane.shape[0]
        s = torch.randn((T, max(E, BLOCKS * BLK)), generator=gen).cuda()
        lists, n = torch.empty((T, K), dtype=torch.int32, device="cuda"), _i32([0] * T)
        cand, cand_n = torch.empty((T, BLOCKS), dtype=torch.int32, device="cuda"), _i32([BLOCKS] * T)
        if mode == "within":
            cand.copy_(torch.stack([torch.randperm(C // BLK, generator=gen)[:BLOCKS].sort().values for _ in range(T)]))
        for split in (False, True):
            work = buffers.split_work(T, E, BLOCKS * BLK, "cuda") if split else None
            ms = _median_ms(partial(_selection, mode, s, lists, n, cand, cand_n, BLK, work, tables))
            assert (n == K).all()
            print(f"selection T={T} C={C} {mode} ratio {ratio} {'split' if split else 'rows'}: {ms:.3f} ms")
