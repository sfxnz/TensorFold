"""DeepSeek's FP8 decode projections on shape-picked tiles: every tile gives ``Mx8Linear``'s bits at 1 to 32 rows."""

from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_dsv41_mx8 import SHAPES, _capture, _linear, _x

from tensorfold.cuda.kernels import qmm
from tensorfold.families.deepseek_v41.cuda import gemv, mx8

ROWS = (*range(1, 25), 31, 32)              # a verify window's 1..24 rows, and the 32-row tile's edge
TILES = [gemv.Tiles(bn, st, fuse) for bn, st, fuse in itertools.product(gemv.COLUMNS, gemv.STAGES, (False, True))]


def _use(monkeypatch, spec: str) -> None:
    monkeypatch.setenv(gemv.KNOB, spec)
    gemv.table.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_table(monkeypatch):
    monkeypatch.delenv(gemv.KNOB, raising=False)
    gemv.table.cache_clear()
    yield
    gemv.table.cache_clear()


@pytest.mark.parametrize("name", SHAPES)
def test_every_tile_gives_the_shared_kernels_bits_at_every_row_count(name):
    lin = _linear(name)
    x = _x(max(ROWS), lin.k, 30)
    for f32 in (False, True):
        want = {m: (mx8._f32(lin, x[:m], None, False) if f32 else lin(x[:m])) for m in ROWS}
        for t, m in itertools.product(TILES, ROWS):
            got = torch.empty((m, lin.n), dtype=torch.float32 if f32 else torch.bfloat16, device="cuda")
            gemv._ext().gemv(x[:m], lin.w8, lin.bs, got, lin.n, qmm.split_k(lin.n, lin.k), lin.npad, t.bn,
                             t.stages, t.fuse, f32)
            assert torch.equal(got, want[m]), (name, t, m, f32)


def test_mm_takes_the_knobs_tiles_for_decode_rows_only(monkeypatch):
    lin, other = _linear("wo_b"), _linear("wq_b")
    calls = []
    ext = gemv._ext()

    class Counting:
        def gemv(self, *a):
            calls.append((a[4], a[7], a[8], a[9]))
            return ext.gemv(*a)

    x, xq = _x(40, lin.k, 31), _x(6, other.k, 32)
    want = [mx8.mm(lin, x[:6], f32=True), mx8.mm(lin, x[:6]), mx8.mm(lin, x, f32=True), mx8.mm(other, xq)]
    monkeypatch.setattr(gemv, "_ext", lambda: Counting())
    _use(monkeypatch, f"{lin.n}x{lin.k}:32/8/fuse")
    got = [mx8.mm(lin, x[:6], f32=True), mx8.mm(lin, x[:6]), mx8.mm(lin, x, f32=True), mx8.mm(other, xq)]
    assert all(torch.equal(g, w) for g, w in zip(got, want))
    assert calls == [(lin.n, 32, 8, True)] * 2                  # 40 rows and an unnamed shape take Mx8Linear's tiles
    calls.clear()
    _use(monkeypatch, "*:16/4/cluster")
    assert torch.equal(mx8.mm(other, xq), want[3]) and calls == [(other.n, 16, 4, False)]
    calls.clear()
    xp = _x(300, lin.k, 33)
    mx8.mm(lin, xp[:6], prompt=True)
    assert calls == []


def test_an_output_buffer_and_strided_rows_keep_the_bits(monkeypatch):
    lin = _linear("shared_d")
    x = _x(6, 2 * lin.k, 34)[:, lin.k:]                         # rows strided, as a column block of a wider buffer
    want = mx8.mm(lin, x, f32=True)
    _use(monkeypatch, "*:16/6/fuse")
    out = torch.empty((6, lin.n), dtype=torch.float32, device="cuda")
    assert mx8.mm(lin, x, out, f32=True) is out and torch.equal(out, want)
    wide = torch.empty((6, 2 * lin.n), dtype=torch.float32, device="cuda")
    mx8.mm(lin, x, wide[:, :lin.n], f32=True)
    assert torch.equal(wide[:, :lin.n], want)


def test_graph_replays_equal_eager(monkeypatch):
    wo_b, wq_b = _linear("wo_b"), _linear("wq_b")
    _use(monkeypatch, f"{wo_b.n}x{wo_b.k}:32/6/fuse,{wq_b.n}x{wq_b.k}:16/8/cluster")
    for r in (1, 4, 6, 24):
        xb, xq = _x(r, wo_b.k, 35), _x(r, wq_b.k, 36)
        yb = torch.empty((r, wo_b.n), dtype=torch.float32, device="cuda")
        yq = torch.empty((r, wq_b.n), dtype=torch.bfloat16, device="cuda")

        def step(xb=xb, yb=yb, xq=xq, yq=yq):
            mx8.mm(wo_b, xb, yb, f32=True)
            mx8.mm(wq_b, xq, yq)

        graph = _capture(step)
        for seed in (37, 38):
            xb.copy_(_x(r, wo_b.k, seed))
            xq.copy_(_x(r, wq_b.k, seed + 1))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(yb, mx8._f32(wo_b, xb, None, False)), r
            assert torch.equal(yq, wq_b(xq)), r


def test_bad_tiles_are_refused_by_the_extension():
    lin = _linear("wo_b")
    x, y = _x(2, lin.k, 39), torch.empty((2, lin.n), dtype=torch.bfloat16, device="cuda")
    sk = qmm.split_k(lin.n, lin.k)
    for bn, stages, rows in ((48, 4, 2), (32, 5, 2), (32, 4, 33)):
        xr = _x(rows, lin.k, 40)
        yr = y if rows == 2 else torch.empty((rows, lin.n), dtype=torch.bfloat16, device="cuda")
        with pytest.raises(RuntimeError):
            gemv._ext().gemv(xr, lin.w8, lin.bs, yr, lin.n, sk, lin.npad, bn, stages, False, False)
    gemv._ext().gemv(x, lin.w8, lin.bs, y, lin.n, sk, lin.npad, 32, 4, False, False)
    assert torch.equal(y, lin(x))
