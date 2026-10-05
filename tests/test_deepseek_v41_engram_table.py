"""DeepSeek-V4.1's Engram rows read by pread equal the file's bytes (the Python thread pool and the native pool alike),
each rank's columns are one byte range, and admission counts the rank's resident scale rows."""

from __future__ import annotations

import json
import os
import struct
import time
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from tensorfold.families.deepseek_v41 import engram_table
from tensorfold.families.deepseek_v41.cuda import split
from tensorfold.families.deepseek_v41.engram_table import Layout, Reader, Table, rank_rows


class Range(NamedTuple):
    """One rank's hash columns of a table: global rows [lo, hi) and their absolute byte ranges in the file."""

    rows: tuple[int, int]
    weight: tuple[int, int]
    scale: tuple[int, int]


def rank_range(layout: Layout, layer: int, rank: int, world: int = 2) -> Range:
    """Rank ``rank``'s contiguous hash columns of ``layer`` (ceil(columns / world) each, rank order)."""

    t = layout.table(layer)
    cols = len(t.bounds) - 1
    per = -(-cols // world)
    lo, hi = t.bounds[min(cols, per * rank)], t.bounds[min(cols, per * (rank + 1))]
    return Range((lo, hi), (t.weight_abs + t.wrow * lo, t.weight_abs + t.wrow * hi),
                 (t.scale_abs + t.srow * lo, t.scale_abs + t.srow * hi))

BUCKETS = ((7, 11, 13, 17, 19, 23, 29, 31), (37, 41, 43, 47, 53, 59, 61, 67))    # 150 and 408 rows
STARTS = (664, 672)                     # the pack's data starts: 152 and 160 past a 256-byte row boundary


def _primes(vocab: int, heads: int, max_ngram: int, layers: int) -> list[list[int]]:
    """Each layer's bucket sizes by the reference's rule: the next unused prime above ``vocab - 1``, per n-gram."""

    def prime(n: int) -> bool:
        return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))

    out, seen = [], set()
    for _ in range(layers):
        sizes = []
        for _ in range(max_ngram - 1):
            current = vocab - 1
            for _ in range(heads):
                current += 1
                while not prime(current) or current in seen:
                    current += 1
                seen.add(current)
                sizes.append(current)
        out.append(sizes)
    return out


def _write(path: Path, tensors: dict[str, tuple[str, np.ndarray]], data: int) -> None:
    """A safetensors file of ``tensors`` (name -> dtype, uint8 array) whose tensor bytes start at ``data``."""

    header, blobs, at = {}, [], 0
    for name, (dtype, array) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [at, at + array.nbytes]}
        blobs.append(array.tobytes())
        at += array.nbytes
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", data - 8) + text.ljust(data - 8) + b"".join(blobs))


@pytest.fixture
def pack(tmp_path):
    """Two Engram tables in two files behind symlinks, as the pack lays them out, and their rows."""

    rng = np.random.default_rng(41)
    store, model = tmp_path / "store", tmp_path / "model"
    store.mkdir(), model.mkdir()
    rows, weight_map = [], {}
    for i, (layer, sizes, data) in enumerate(zip((1, 14), BUCKETS, STARTS)):
        n = sum(sizes)
        w = rng.integers(0, 256, (n, 256), dtype=np.uint8)
        s = rng.integers(0, 256, (n, 8), dtype=np.uint8)
        name = f"model-0000{i + 1}-of-00002.safetensors"
        _write(store / name, {f"layers.{layer}.engram.embed.weight": ("F8_E4M3", w),
                              f"layers.{layer}.engram.embed.scale": ("F8_E8M0", s),
                              f"layers.{layer}.engram.q_weight": ("BF16", np.zeros((4, 6), np.uint8))}, data)
        (model / name).symlink_to(store / name)
        weight_map.update({f"layers.{layer}.engram.embed.{p}": name for p in ("weight", "scale")})
        rows.append((w, s))
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return model, np.concatenate([w for w, _ in rows]), np.concatenate([s for _, s in rows])


def _reader(model: Path) -> Reader:
    return Reader(Layout.read(model, (1, 14), BUCKETS), workers=4)


def test_rank_ranges_equal_the_pack_tables_byte_offsets():
    """The pack format: each rank's 12 hash columns are one weight range and one scale range in the file."""

    want = {(1, 0): ((664, 49_152_446_104), (98_305_579_672, 99_841_593_592)),
            (1, 1): ((49_152_446_104, 98_305_579_672), (99_841_593_592, 101_377_629_016)),
            (14, 0): ((672, 49_153_796_768), (98_308_271_264, 99_844_327_392)),
            (14, 1): ((49_153_796_768, 98_308_271_264), (99_844_327_392, 101_380_404_720))}
    primes = _primes(16_000_000, 8, 4, 2)
    tables = []
    for layer, rows, data, sizes in zip((1, 14), (384_006_168, 384_016_682), STARTS, primes):
        header = {"__metadata__": {"format": "pt"},
                  f"layers.{layer}.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [rows, 256],
                                                          "data_offsets": [0, rows * 256]},
                  f"layers.{layer}.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [rows, 8],
                                                         "data_offsets": [rows * 256, rows * 264]}}
        tables.append(Table.parse(layer, Path(f"layer{layer}"), header, data, data + rows * 264, sizes))
    layout = Layout(tuple(tables), (), (0, tables[0].rows, tables[0].rows + tables[1].rows))
    assert layout.table(1).bounds[12] == 192_001_740 and layout.table(14).bounds[12] == 192_007_016
    for (layer, rank), (weight, scale) in want.items():
        got = rank_range(layout, layer, rank)
        assert (got.weight, got.scale) == (weight, scale)
        assert got.rows == ((0, 192_001_740), (192_001_740, 384_006_168), (0, 192_007_016),
                            (192_007_016, 384_016_682))[2 * (layer == 14) + rank]
    assert rank_range(layout, 1, 0, world=1).rows == (0, 384_006_168)


def test_rows_by_global_id_are_byte_exact_through_symlinks_and_page_straddles(pack):
    model, weights, scales = pack
    layout = Layout.read(model, (1, 14), BUCKETS)
    assert [t.weight_abs for t in layout.tables] == list(STARTS) and layout.starts == (0, 150, 558)
    assert all(t.path.parent.name == "store" for t in layout.tables)
    ids = np.array([0, 149, 150, 557, 13, 13, 300, 14, 0, 29, 46], dtype=np.int64)
    straddle = [(STARTS[i >= 150] + 256 * (i - 150 * (i >= 150))) % 4096 > 4096 - 256 for i in ids.tolist()]
    assert any(straddle) and not all(straddle)
    buf = np.full((ids.size, 264), 7, dtype=np.uint8)      # rows land in strided views of one staging buffer
    reader = Reader(layout, workers=3)
    reader.gather(ids, buf[:, :256], buf[:, 256:])
    reader.close()
    assert buf[:, :256].tobytes() == weights[ids].tobytes() and buf[:, 256:].tobytes() == scales[ids].tobytes()


def test_duplicates_are_read_once_and_short_reads_are_finished(pack, monkeypatch):
    model, weights, scales = pack
    reader = _reader(model)
    calls, real = [], os.pread

    def pread(fd: int, size: int, offset: int) -> bytes:
        calls.append(offset)
        return real(fd, min(size, 33), offset)

    monkeypatch.setattr(os, "pread", pread)
    ids = np.array([5, 400, 5, 5, 400, 151], dtype=np.int64)
    w, s = np.empty((6, 256), np.uint8), np.empty((6, 8), np.uint8)
    reader.gather(ids, w, s)
    reader.close()
    assert w.tobytes() == weights[ids].tobytes() and s.tobytes() == scales[ids].tobytes()
    assert len(calls) == 3 * (256 // 33 + 1) + 3        # 3 distinct rows: 8 pieces of each weight row, 1 scale


def test_gather_async_returns_a_future_of_the_same_bytes(pack):
    model, weights, scales = pack
    reader = _reader(model)
    ids = np.random.default_rng(3).integers(0, 558, 500)
    w, s = np.empty((500, 256), np.uint8), np.empty((500, 8), np.uint8)
    future = reader.gather_async(ids, w, s)
    assert future.result() is None
    assert w.tobytes() == weights[ids].tobytes() and s.tobytes() == scales[ids].tobytes()
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.gather_async(ids, w, s)


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="needs posix_fadvise")
def test_advise_asks_for_merged_pages_and_changes_no_bytes(pack, monkeypatch):
    model, weights, scales = pack
    reader = _reader(model)
    spans, real = [], os.posix_fadvise
    monkeypatch.setattr(os, "posix_fadvise", lambda fd, offset, size, advice: spans.append((fd, offset, size, advice)))
    ids = np.array([0, 1, 2, 557, 300, 151], dtype=np.int64)
    reader.advise(ids)
    reader.advise(np.array([], dtype=np.int64))
    monkeypatch.setattr(os, "posix_fadvise", real)
    assert {advice for *_, advice in spans} == {os.POSIX_FADV_WILLNEED}
    assert all(offset % 4096 == 0 and size % 4096 == 0 and size > 0 for _, offset, size, _ in spans)
    for fd in {fd for fd, *_ in spans}:
        mine = sorted((o, o + n) for f, o, n, _ in spans if f == fd)
        assert all(a[1] < b[0] for a, b in pairwise(mine))      # merged: no two spans touch
    for i in ids.tolist():
        t = reader.layout.tables[i >= 150]
        row = i - reader.layout.starts[i >= 150]
        for at, width in ((t.weight_abs + 256 * row, 256), (t.scale_abs + 8 * row, 8)):
            assert any(o <= at and at + width <= o + n for _, o, n, _ in spans)
    reader.advise(ids)
    w, s = np.empty((6, 256), np.uint8), np.empty((6, 8), np.uint8)
    reader.gather(ids, w, s)
    reader.close()
    assert w.tobytes() == weights[ids].tobytes() and s.tobytes() == scales[ids].tobytes()


def test_a_truncated_file_is_refused_at_layout_read(pack):
    model = pack[0]
    target = (model / "model-00002-of-00002.safetensors").resolve()
    os.truncate(target, target.stat().st_size - 24 - 1)        # into the last scale row (q_weight is 24 bytes)
    with pytest.raises(ValueError, match="past the file"):
        Layout.read(model, (1, 14), BUCKETS)


def test_a_file_cut_short_after_opening_fails_the_gather(pack):
    model = pack[0]
    reader = _reader(model)
    target = (model / "model-00001-of-00002.safetensors").resolve()
    os.truncate(target, 664 + 256 * 100)
    w, s = np.empty((1, 256), np.uint8), np.empty((1, 8), np.uint8)
    with pytest.raises(OSError, match="the checkpoint changed"):
        reader.gather(np.array([120]), w, s)
    reader.close()


def test_digest_follows_each_file_size_and_header(pack):
    model = pack[0]
    first = Layout.read(model, (1, 14), BUCKETS).digest()
    assert Layout.read(model, (1, 14), BUCKETS).digest() == first and isinstance(first, int)
    target = (model / "model-00002-of-00002.safetensors").resolve()
    raw = bytearray(target.read_bytes())
    head = raw.index(b'"BF16"')
    raw[head:head + 6] = b'"F16" '
    target.write_bytes(bytes(raw))
    assert Layout.read(model, (1, 14), BUCKETS).digest() != first


def test_bad_buckets_ids_and_buffers_are_refused(pack):
    model = pack[0]
    with pytest.raises(ValueError, match="hash buckets cover"):
        Layout.read(model, (1, 14), (BUCKETS[0], BUCKETS[1][:-1]))
    with pytest.raises(ValueError, match="bucket lists"):
        Layout.read(model, (1,), BUCKETS)
    reader = _reader(model)
    w, s = np.empty((1, 256), np.uint8), np.empty((1, 8), np.uint8)
    with pytest.raises(ValueError, match=r"\[0, 558\)"):
        reader.gather(np.array([558]), w, s)
    with pytest.raises(TypeError):
        reader.gather(np.array([1.0]), w, s)
    with pytest.raises(ValueError, match="buffer"):
        reader.gather(np.array([1, 2]), w, s)
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.gather(np.array([1]), w, s)


@pytest.mark.skipif(not os.environ.get("TF_DSV41_MODEL"), reason="set TF_DSV41_MODEL to the checkpoint")
def test_real_rows_equal_the_memory_map(capsys):
    """256 random rows per rank and layer equal numpy.memmap slices; prints the read rate (a receipt, not a gate)."""

    model = Path(os.environ["TF_DSV41_MODEL"])
    text = json.loads((model / "config.json").read_text())
    text = text.get("text_config", text)
    layers = tuple(text["engram_layer_ids"])
    primes = _primes(text["engram_vocab_size"], text["engram_n_heads"], text["engram_max_ngram_size"], len(layers))
    layout = Layout.read(model, layers, primes)
    assert [t.rows for t in layout.tables] == list(text["engram_num_embeddings"])
    rng = np.random.default_rng(int(time.time()))
    ids = []
    for i, layer in enumerate(layers):
        for rank in (0, 1):
            lo, hi = rank_range(layout, layer, rank).rows
            ids.append(layout.starts[i] + rng.integers(lo, hi, 256))
    ids = np.concatenate(ids)
    reader = Reader(layout)
    w, s = np.empty((ids.size, 256), np.uint8), np.empty((ids.size, 8), np.uint8)
    rates = []
    for _ in range(2):
        start = time.perf_counter()
        reader.gather(ids, w, s)
        rates.append(ids.size / (time.perf_counter() - start))
    reader.close()
    for i, t in enumerate(layout.tables):
        mine = (ids >= layout.starts[i]) & (ids < layout.starts[i + 1])
        rows = ids[mine] - layout.starts[i]
        wmap = np.memmap(t.path, np.uint8, "r", offset=t.weight_abs, shape=(t.rows, t.wrow))
        smap = np.memmap(t.path, np.uint8, "r", offset=t.scale_abs, shape=(t.rows, t.srow))
        assert w[mine].tobytes() == wmap[rows].tobytes() and s[mine].tobytes() == smap[rows].tobytes()
    with capsys.disabled():
        print(f"\nEngram rows/s ({ids.size} rows, {engram_table.WORKERS} workers): "
              f"first {rates[0]:,.0f}, repeat {rates[1]:,.0f}")


def test_rank_rows_are_each_ranks_hash_columns():
    primes = _primes(16_000_000, 8, 4, 2)
    assert [rank_rows(p, r, 2) for p in primes for r in (0, 1)] == [
        (0, 192_001_740), (192_001_740, 384_006_168), (0, 192_007_016), (192_007_016, 384_016_682)]
    assert rank_rows(primes[0], 0, 1) == (0, 384_006_168)
    for sizes in BUCKETS:
        bounds = np.cumsum([0, *sizes])
        assert [rank_rows(sizes, r, 2) for r in (0, 1)] == [(0, bounds[4]), (bounds[4], bounds[8])]
    with pytest.raises(ValueError, match="even share"):
        rank_rows(BUCKETS[0][:7], 0, 2)
    with pytest.raises(ValueError, match="even share"):
        rank_rows(BUCKETS[0], 2, 2)


def test_resident_scale_rows_are_counted_for_admission():
    """2.86 GiB of scale rows per rank; the weight bytes and, without the rows, the tables count nothing."""

    rows = {r: {layer: hi - lo for layer, (lo, hi) in zip((1, 14), (rank_rows(p, r, 2) for p in
                                                                         _primes(16_000_000, 8, 4, 2)))}
            for r in (0, 1)}
    infos = {f"layers.{layer}.engram.embed.{part}": {"dtype": dtype, "shape": [n, width]}
             for layer, n in ((1, 384_006_168), (14, 384_016_682))
             for part, dtype, width in (("weight", "F8_E4M3", 256), ("scale", "F8_E8M0", 8))}
    for r, want in ((0, 3_072_070_048), (1, 3_072_112_752)):
        sizes = [split.weights_estimate(name, info, engram_rows=rows[r]) for name, info in infos.items()]
        assert sum(size for size, _ in sizes) == want and all(mapped == 0 for _, mapped in sizes)
        assert 2.86 < want / 2**30 < 2.87
    assert all(split.weights_estimate(name, info) == (0, 0) for name, info in infos.items())


def test_weight_bytes_alone_skip_the_scale_reads(pack, monkeypatch):
    model, weights, _ = pack
    reader = _reader(model)
    calls, real = [], os.pread

    def pread(fd: int, size: int, offset: int) -> bytes:
        calls.append(size)
        return real(fd, size, offset)

    monkeypatch.setattr(os, "pread", pread)
    ids = np.array([5, 400, 5, 151, 557], dtype=np.int64)
    w = np.full((5, 256), 9, np.uint8)
    reader.gather(ids, w)
    reader.close()
    assert w.tobytes() == weights[ids].tobytes() and calls == [256] * 4


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="needs posix_fadvise")
def test_advise_without_scales_asks_for_the_weight_pages_only(pack, monkeypatch):
    model = pack[0]
    reader = _reader(model)
    spans = []
    monkeypatch.setattr(os, "posix_fadvise", lambda fd, offset, size, advice: spans.append((fd, offset, size)))
    ids = np.array([0, 1, 2, 557, 300, 151], dtype=np.int64)
    reader.advise(ids, scales=False)
    monkeypatch.undo()
    fds = {fd: t.path for fd, t in zip(reader._fds, reader.layout.tables)}
    reader.close()
    pages = set()
    for i in ids.tolist():
        t = reader.layout.tables[i >= 150]
        at = t.weight_abs + 256 * (i - reader.layout.starts[i >= 150])
        pages |= {(t.path, p) for p in range(at // 4096, (at + 255) // 4096 + 1)}
    got = {(fds[fd], p) for fd, offset, size in spans for p in range(offset // 4096, (offset + size) // 4096)}
    assert got == pages


def _native(workers: int):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the native pool builds through tensorfold.cuda.build, which needs a GPU")
    from tensorfold.families.deepseek_v41.cuda import engram

    return engram.read_pool(workers)


@pytest.mark.parametrize("workers", [1, 3, 32])
def test_native_pool_reads_the_python_pools_bytes(pack, workers):
    model, weights, scales = pack
    layout = Layout.read(model, (1, 14), BUCKETS)
    native, python = Reader(layout, pool=_native(workers)), Reader(layout, workers=4)
    rng = np.random.default_rng(workers)
    for n, cold in ((0, False), (1, True), (7, False), (5000, True), (5000, False)):
        ids = rng.integers(0, 558, n)
        if cold and hasattr(os, "posix_fadvise"):          # out of the page cache: the pool's disk reads
            for t in layout.tables:
                fd = os.open(t.path, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
        got = [np.full((n, 256), 1, np.uint8), np.full((n, 8), 2, np.uint8)]
        want = [np.full((n, 256), 3, np.uint8), np.full((n, 8), 4, np.uint8)]
        native.gather(ids, *got)
        python.gather(ids, *want)
        assert got[0].tobytes() == want[0].tobytes() == weights[ids].tobytes()
        assert got[1].tobytes() == want[1].tobytes() == scales[ids].tobytes()
        alone = np.empty((n, 256), np.uint8)
        native.gather_async(ids, alone).result()
        assert alone.tobytes() == weights[ids].tobytes()
    native.close()
    python.close()


def test_native_pool_fails_a_file_cut_short(pack):
    model = pack[0]
    reader = Reader(Layout.read(model, (1, 14), BUCKETS), pool=_native(4))
    os.truncate((model / "model-00001-of-00002.safetensors").resolve(), 664 + 256 * 100 + 17)
    w = np.empty((3, 256), np.uint8)
    with pytest.raises(OSError, match="byte 26281: the checkpoint changed"):     # 17 bytes into row 100
        reader.gather(np.array([3, 100, 99]), w)
    reader.gather(np.array([3, 99]), w[:2])
    with pytest.raises(OSError, match="the checkpoint changed"):               # the scale rows are past the end
        reader.gather(np.array([3]), w[:1], np.empty((1, 8), np.uint8))
    reader.close()
