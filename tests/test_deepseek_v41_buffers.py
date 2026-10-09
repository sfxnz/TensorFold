"""DeepSeek-V4.1-Flash's per-rank state and scratch: shapes on the meta device, byte totals, snapshot row views."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import buffers, weights
from tensorfold.families.deepseek_v41.cuda.lanes import MAX_LANES, Lanes

CFG = Config.read(Path(__file__).parent / "fixtures" / "deepseek_v41")
CAPACITIES = (4096, 65536, 1 << 20)
BF, F32, I32, U8 = torch.bfloat16, torch.float32, torch.int32, torch.uint8


def _walk(obj) -> int:
    """Every tensor reachable from ``obj``'s attributes, counted once, independently of the module's own walk."""

    seen, total, stack = set(), 0, [obj]
    while stack:
        v = stack.pop()
        if isinstance(v, torch.Tensor):
            if id(v) not in seen:
                seen.add(id(v))
                total += v.numel() * v.element_size()
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        elif isinstance(v, dict):
            stack.extend(v.values())
        elif hasattr(v, "__dict__") and type(v).__module__.startswith("tensorfold."):
            stack.extend(vars(v).values())
    return total


@pytest.mark.parametrize("capacity", CAPACITIES)
def test_state_shapes(capacity):
    st = buffers.State(CFG, capacity, "meta")
    assert st.rings.shape == (43, 128, 528) and st.rings.dtype == U8          # FP8 codes, an E8M0 byte a 32
    assert {k: tuple(v.shape) for k, v in st.comp.items()} == {                # FP4 codes, an e4m3 byte a 16
        2: (capacity // 2 + 1, 288), 8: (capacity // 2 + 1, 288), 14: (capacity // 2 + 1, 288), 20: (capacity, 288)}
    assert {k: tuple(v.shape) for k, v in st.index_k.items()} == {             # FP4 codes, an E8M0 byte a 32
        2: (capacity // 2 + 1, 68), 8: (capacity // 2 + 1, 68), 14: (capacity // 2 + 1, 68), 20: (capacity, 68)}
    assert all(v.dtype == U8 for v in [*st.comp.values(), *st.index_k.values()])
    assert st.pooled == (2, 8, 14)
    assert st.tail.shape == (3, 2, 512) and st.tail.dtype == F32
    assert st.tail_valid.shape == (3,) and st.tail_valid.dtype == I32
    assert st.pos_dev.shape == (1,) and st.pos_dev.dtype == I32
    fixed = 43 * 128 * 528 + 3 * 2 * 512 * 4 + 3 * 4 + 4 + 3 * 356         # rings, tails, flags, pos, entry rounding
    assert st.nbytes() == _walk(st) == 890 * capacity + fixed


def test_state_position_and_reset():
    st = buffers.State(CFG, 64, "cpu")
    st.set_pos(37)
    assert st.pos == 37 and int(st.pos_dev) == 37
    st.tail_valid.fill_(1)
    st.history = [5, 6, 7]
    st.reset()
    assert st.pos == 0 and int(st.pos_dev) == 0 and not st.tail_valid.any() and st.history == []


@pytest.mark.parametrize("n", [0, 1, 2, 7, 127, 128, 129, 2047, 4095])
def test_row_views(n):
    st = buffers.State(CFG, 4096, "meta")
    views = st.row_views(n)
    assert [tuple(v.shape) for v in views] == [(n // 2 + 1, 288), (n // 2 + 1, 68)] * 3 + [(n + 1, 288),
                                                                                       (n + 1, 68)]
    size = sum(v.numel() * v.element_size() for v in views)
    assert size == 3 * (n // 2 + 1) * 356 + (n + 1) * 356
    assert 0 < size - 890 * n <= 5 * 356                        # entry rounding: at most one entry a cache, plus one
    assert all(v._base is not None for v in views)              # views of the live caches, not copies


def test_row_views_stop_at_the_cache():
    st = buffers.State(CFG, 4096, "meta")
    assert [v.shape[0] for v in st.row_views(4096)] == [2049, 2049] * 3 + [4096, 4096]


def _expected(rows: int, capacity: int, prefill: bool) -> dict:
    """The buffer shapes for the pack's config on two ranks."""

    taps = min(rows, 129 if prefill else 128)     # a chunk's also the row before a window: a snapshot at its last row
    scored = min(rows, buffers.score_rows(capacity)) if prefill else rows
    out = {
        "ids": ((rows,), I32), "X": ((rows, 4, 5120), BF), "pre_in": ((rows, 4), F32),
        "hcpart": ((rows, 16, 32), F32), "pre_a": ((rows, 4), F32), "post_a": ((rows, 4), F32),
        "comb_a": ((rows, 16), F32), "pre_f": ((rows, 4), F32), "post_f": ((rows, 4), F32),
        "comb_f": ((rows, 16), F32), "xn": ((rows, 5120), BF),
        "qakv": ((rows, 1792), BF), "qr": ((rows, 1280), BF), "q": ((rows, 32, 512), BF), "o": ((rows, 32, 512), BF),
        "u": ((rows, 4096), BF), "kvw": ((43, rows, 528), U8),
        "cmp": ((3, rows, 2, 512), F32), "lat": ((rows, 512), BF), "latp": ((rows, 288), U8),
        "kI": ((rows, 128), BF), "kIp": ((rows, 68), U8),
        "epos": ((rows,), I32), "qI": ((rows, 32, 128), BF),
        "wI": ((rows, 32), F32), "scores": ((scored, capacity), F32), "cand": ((rows, 2048), I32),
        "cand_n": ((rows,), I32), "lists": ((rows, 512), I32), "list_n": ((rows,), I32),
        "mlog": ((rows, 384), F32), "pick": ((rows, 6), I32), "wts": ((rows, 6), F32),
        "sgu": ((rows, 2304), BF), "sact": ((rows, 1152), BF), "sd": ((rows, 5120), F32),
        "part": ((rows, 5120), F32), "gath": ((2, rows, 5120), F32),
        "eraw": ((rows, 2, 12, 264), U8), "eidx": ((rows, 2, 12), torch.int64), "eloc": ((rows, 2, 3072), BF),
        "egat": ((2, rows, 2, 3072), BF),
        "eng": ((rows, 2, 6144), BF), "ekv": ((rows, 12800), BF), "ekv_gat": ((2, rows, 12800), BF),
        "taps": ((taps, 3, 5120), BF), "hidden": ((rows, 5120), BF), "fnormed": ((rows, 5120), BF),
        "logits": ((1 if prefill else rows, 64640), F32), "mx": ((taps, 5120), BF), "mx_gat": ((2, taps, 2560), BF),
    }
    if not prefill:
        out.update({
            "attn_part": ((rows, 32, 5, 514), F32),
            "xd": ((5, 4, 5120), BF), "bkv": ((5, 512), BF), "dmlog": ((5, 128), F32), "dpick": ((5, 4), I32),
            "dwts": ((5, 4), F32), "dact": ((5, 4, 1152), BF), "dey": ((5, 4, 5120), F32),
            "dlog": ((5, 64640), F32), "me": ((5, 256), BF), "conf": ((5,), F32),
        })
    return out


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("rows,prefill", [(6, False), (2048, True)])
def test_buffer_shapes(rows, prefill, capacity):
    b = buffers.Buffers(CFG, 2, rows, capacity, prefill, device="meta")
    for name, (shape, dtype) in _expected(rows, capacity, prefill).items():
        t = getattr(b, name)
        assert (tuple(t.shape), t.dtype) == (shape, dtype), name
    assert b.exl3.rows == (1024 if prefill else rows) and b.exl3.slots == 6       # no shared-expert slot (D10)
    assert b.exl3.y.shape == (b.exl3.rows * 6, 5120)
    if prefill:
        assert b.attn_part is None and not hasattr(b, "dplan") and not hasattr(b, "split")
        assert b.scores.numel() * 4 == min(1 << 30, 2048 * capacity * 4)
    else:
        assert (b.dplan.rows, b.dplan.slots, b.dplan.experts) == (5, 4, 129)
        assert {k: (tuple(v.shape), v.dtype) for k, v in b.split.items()} == {
            "hist": ((rows, 2048), I32), "bn": ((rows,), I32), "bkey": ((rows, capacity), I32),
            "bitem": ((rows, capacity), I32), "counts": ((rows, 3, -(-max(capacity, 16384) // 1024)), I32),
            "thr": ((rows, 2), I32)}
    assert b.ids_host is None and b.staged is None and b.eraw_host == b.eraw_done == b.eidx_host == []  # no pinning


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("rows,prefill", [(6, False), (2048, True)])
def test_bytes_is_the_allocation(rows, prefill, capacity):
    b = buffers.Buffers(CFG, 2, rows, capacity, prefill, device="meta")
    assert buffers.bytes(CFG, 2, rows, capacity, prefill) == b.nbytes() == _walk(b)


def test_prompt_byte_figures():
    b = buffers.Buffers(CFG, 2, 2048, 1 << 20, True, device="meta")
    assert b.kvw.numel() == 43 * 2048 * 528                          # 46 MB
    assert 490e6 < _walk(b.exl3) < 500e6                             # 1024 rows x 6 slots, no sentinel (D10)
    assert b.scores.numel() * 4 == 1 << 30


def test_score_rows():
    assert buffers.score_rows(1 << 20) == 256
    assert buffers.score_rows(65536) == 2048
    assert buffers.score_rows(3_000_000) == 80
    assert buffers.score_rows(1 << 30) == 16
    assert all(buffers.score_rows(e) % 16 == 0 for e in (1, 1000, 123_457, 999_999))


def test_one_rank_holds_whole_tensors():
    b = buffers.Buffers(CFG, 1, 6, 4096, device="meta")
    assert b.q.shape == (6, 64, 512) and b.logits.shape == (6, 129280) and b.eraw.shape == (6, 2, 24, 264)
    assert b.gath.shape == (1, 6, 5120) and b.sact.shape == (6, 2304)


def test_reduced_layer_set():
    cfg = Config.read(Path(__file__).parent / "fixtures" / "deepseek_v41", layers=8)
    st = buffers.State(cfg, 4096, "meta")
    assert st.rings.shape[0] == 11 and list(st.comp) == [2] and st.pooled == (2,)
    b = buffers.Buffers(cfg, 2, 6, 4096, device="meta")
    assert b.kvw.shape == (11, 6, 528) and b.cmp.shape == (1, 6, 2, 512) and b.scores.shape == (6, 2049)
    assert b.eraw.shape == (6, 1, 12, 264)


@pytest.mark.parametrize("slots", range(1, MAX_LANES + 1))
def test_lanes_stack_a_state_per_lane(slots):
    lanes, one = Lanes(CFG, slots, 65536, "meta"), buffers.State(CFG, 65536, "meta")
    assert lanes.rings.shape == (slots, 43, 128, 528) and lanes.tail.shape == (slots, 3, 2, 512)
    assert {k: tuple(v.shape) for k, v in lanes.comp.items()} == {k: (slots, *v.shape) for k, v in one.comp.items()}
    assert {k: tuple(v.shape) for k, v in lanes.index_k.items()} == {
        k: (slots, *v.shape) for k, v in one.index_k.items()}
    assert lanes.tail_valid.shape == (slots, 3) and lanes.pos_dev.shape == (slots,)
    assert lanes.nbytes() == slots * one.nbytes() == slots * lanes.view(slots - 1).nbytes()


def test_lanes_bound():
    with pytest.raises(ValueError, match="slots"):
        Lanes(CFG, MAX_LANES + 1, 4096, "meta")


def test_a_lane_view_aliases_the_stack():
    lanes = Lanes(CFG, 3, 64, "cpu")
    v = lanes.view(1)
    assert v is lanes.view(1) and v.pos == 0 and v.history == [] and v.capacity == 64 and v.pooled == (2, 8, 14)
    assert v.rings.data_ptr() == lanes.rings[1].data_ptr() and v.rings.is_contiguous()
    v.set_pos(37)
    for t in (v.rings, v.tail, v.tail_valid, *v.comp.values(), *v.index_k.values()):
        t.fill_(7)
    assert lanes.pos_dev.tolist() == [0, 37, 0] and lanes.view(0).pos == lanes.view(2).pos == 0
    for t in (lanes.rings, lanes.tail, lanes.tail_valid, *lanes.comp.values(), *lanes.index_k.values()):
        assert (t[1] == 7).all() and not t[0].any() and not t[2].any()
    v.reset()
    assert int(lanes.pos_dev[1]) == 0 and not lanes.tail_valid[1].any()


@pytest.mark.parametrize("slots", range(1, MAX_LANES + 1))
def test_lane_tables_add_only_their_bytes(slots):
    rows = 6 * slots
    solo, b = buffers.Buffers(CFG, 2, rows, 65536, device="meta"), buffers.Buffers(CFG, 2, rows, 65536,
                                                                                    device="meta", lanes=slots)
    assert not hasattr(solo, "lane") and not hasattr(solo, "segs")
    assert {n: (tuple(getattr(b, n).shape), getattr(b, n).dtype) for n in ("lane", "rpos", "seg0", "segs", "nseg")} \
        == {"lane": ((rows,), I32), "rpos": ((rows,), I32), "seg0": ((rows,), I32), "segs": ((slots, 2), I32),
            "nseg": ((1,), I32)}
    assert b.lane_host is b.rpos_host is b.seg0_host is b.segs_host is b.nseg_host is None
    assert b.nbytes() - solo.nbytes() == 4 * (3 * rows + 2 * slots + 1) and b.nbytes() == _walk(b)
    assert solo.nbytes() == buffers.bytes(CFG, 2, rows, 65536)


@pytest.mark.parametrize("slots", (1, MAX_LANES))
def test_greedy_scratch_adds_only_its_keys(slots):
    """GREEDY_ENV's decode buffers hold each row's greedy key (this rank's, every rank's, the least); none else do."""

    from tensorfold.families.deepseek_v41.cuda.sample import GREEDY_ENV, greedy_on_device

    assert all(greedy_on_device(env) for env in ({}, {GREEDY_ENV: "1"}, {GREEDY_ENV: " 1 "}))
    assert not greedy_on_device({GREEDY_ENV: "0"}) and not greedy_on_device({GREEDY_ENV: " 0 "})
    rows = 6 * slots
    off, on = (buffers.Buffers(CFG, 2, rows, 65536, device="meta", lanes=slots, greedy=g) for g in (False, True))
    assert off.gkeys is off.ggot is off.gbest is off.gkeys_host is None
    assert {n: (tuple(getattr(on, n).shape), getattr(on, n).dtype) for n in ("gkeys", "ggot", "gbest")} == {
        "gkeys": ((rows,), torch.int64), "ggot": ((2 * rows,), torch.int64), "gbest": ((rows,), torch.int64)}
    assert on.gkeys_host is None                                     # no pinning
    assert on.nbytes() - off.nbytes() == 8 * 4 * rows and on.nbytes() == _walk(on)
    assert buffers.Buffers(CFG, 2, 2048, 4096, True, device="meta", greedy=True).gkeys is None   # a prompt chunk


def test_weights_fields():
    assert [f.name for f in dataclasses.fields(weights.Weights)] == [
        "cfg", "rank", "world", "comm", "device", "vocab_offset", "embed", "layers", "norm", "head", "dspark",
        "engram", "rope", "engram_scales"]
    assert issubclass(weights.StageW, weights.LayerW)
