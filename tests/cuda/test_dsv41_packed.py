"""Packed caches: packed rows dequantize to the QDQ's bf16 bits, and attention and index scores over packed rows equal
those over the bf16 rows bit for bit."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import triton.language as tl

from tensorfold.families.deepseek_v41.cuda import MAX_ROWS, quant
from tensorfold.families.deepseek_v41.cuda import attn_kernel as ak
from tensorfold.families.deepseek_v41.cuda import indexer as ix

DEV = "cuda"
KINDS = {"fp8": (quant.fp8_qdq_1x32, quant.FP8), "fp4_e8m0": (quant.fp4_qdq_1x32_e8m0, quant.FP4_E8M0),
         "fp4_e4m3": (quant.fp4_qdq_1x16_e4m3, quant.FP4_E4M3)}


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16).cpu()


def _packed(kind: int, x: torch.Tensor) -> torch.Tensor:
    fn = next(f for f, k in KINDS.values() if k == kind)
    out = torch.full((*x.shape[:-1], quant.width(x.shape[-1], kind)), 0xA5, dtype=torch.uint8, device=x.device)
    assert fn(x, out) is out
    return out


@triton.jit
def _unpack_rows(P, OUT, sp, KIND: tl.constexpr, N: tl.constexpr, D: tl.constexpr):
    rows = tl.program_id(0) * N + tl.arange(0, N)
    at = P + rows.to(tl.int64) * sp
    if KIND == 0:
        v = quant.unpack_fp8(at, rows >= 0, N, D)
    else:
        v = quant.unpack_fp4(at, rows >= 0, N, D, 32 if KIND == 1 else 16)
    tl.store(OUT + rows[:, None] * D + tl.arange(0, D)[None, :], v.to(tl.bfloat16))


def _values(seed: int) -> torch.Tensor:
    """Rows of 512: random at scales over the bf16 range, extremes, signed zeros, and every positive bf16 value
    heading 16 values (every power-of-two and e4m3 scale boundary)."""

    g = torch.Generator().manual_seed(seed)
    big, sub = torch.finfo(torch.bfloat16).max, 2.0**-133
    rows = [torch.randn((4, 512), generator=g) * scale for scale in (1e-30, 1e-3, 1.0, 37.0, 1e4, 1e30)]
    rows.append(torch.tensor([[0.0] * 32, [-0.0] * 32, [big] + [-big] * 31, [big, sub] * 16, [sub, -sub] * 16,
                              [1e-4, -1e-5] * 16, [6 * 2.0**-126, 0.0] * 16, [6 * 2.0**-9, -0.0] * 16,
                              [448.0, 1e-30] * 16, [2688.0, -2687.0] * 16, [-big, -0.0] * 16, [5e4, -3e-3] * 16,
                              [1.0, -0.2] * 16, [0.0, -0.2] * 16, [3.0, 0.75] * 16, [6.0, -5.0] * 16]).view(-1, 512))
    amax = torch.arange(1, 0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    groups = torch.cat([amax[:, None], amax[:, None] * (torch.rand((len(amax), 15), generator=g) * 2 - 1)], 1)
    rows.append(torch.cat([groups.flatten(), groups.new_zeros(-groups.numel() % 512)]).view(-1, 512))
    return torch.cat(rows).to(torch.bfloat16)


@pytest.mark.parametrize("name", KINDS)
def test_packed_rows_dequantize_to_the_qdq_bits(name):
    """Host ``unpack`` and the kernels' ``unpack_fp8``/``unpack_fp4`` give the bf16 QDQ, -0 and every scale
    included."""

    fn, kind = KINDS[name]
    x = _values(1).to(DEV)
    if kind == quant.FP4_E8M0:
        x = x.view(-1, 128)                      # the index key width
    want = fn(x.clone())
    rows = _packed(kind, x)
    assert torch.equal(_bits(quant.unpack(rows, kind, x.shape[1])), _bits(want))
    n = x.shape[0] - x.shape[0] % 16
    out = torch.empty((n, x.shape[1]), dtype=torch.bfloat16, device=DEV)
    _unpack_rows[(n // 16,)](rows, out, rows.stride(0), KIND=kind, N=16, D=x.shape[1])
    assert torch.equal(_bits(out), _bits(want[:n]))


def test_packed_rows_do_not_depend_on_the_window():
    x = (torch.randn((MAX_ROWS, 512), device=DEV) * 3).to(torch.bfloat16)
    for _, kind in KINDS.values():
        alone = [_packed(kind, x[r:r + 1]) for r in range(MAX_ROWS)]
        for rows in range(2, MAX_ROWS + 1):
            assert torch.equal(_packed(kind, x[:rows]), torch.cat(alone[:rows]))


def test_packing_refuses_other_shapes():
    x = torch.zeros((3, 512), dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError):
        quant.fp8_qdq_1x32(x, torch.empty((3, 512), dtype=torch.uint8, device=DEV))
    with pytest.raises(ValueError):
        quant.fp4_qdq_1x16_e4m3(x, torch.empty((3, quant.width(512, quant.FP4_E8M0)), dtype=torch.uint8, device=DEV))
    with pytest.raises(ValueError):
        quant.unpack(torch.empty((3, 288), dtype=torch.uint8), quant.FP8, 512)


class Caches:
    """One layer's window KV (FP8 grid) and compressed entries (FP4 grid), bf16 and packed, and ascending lists."""

    def __init__(self, n: int, seed: int) -> None:
        g = torch.Generator(device=DEV).manual_seed(seed)
        kv = torch.randn((n, 512), generator=g, device=DEV) * torch.rand((n, 1), generator=g, device=DEV) * 4
        comp = torch.randn((n + 1, 512), generator=g, device=DEV) * torch.rand((n + 1, 1), generator=g, device=DEV)
        kv, comp = kv.to(torch.bfloat16), comp.to(torch.bfloat16)
        self.kv, self.kv8 = quant.fp8_qdq_1x32(kv.clone()), _packed(quant.FP8, kv)
        self.comp, self.comp4 = quant.fp4_qdq_1x16_e4m3(comp.clone()), _packed(quant.FP4_E4M3, comp)
        self.q = (2 * torch.randn((n, 32, 512), generator=g, device=DEV)).to(torch.bfloat16)
        self.sink = torch.randn(32, generator=g, device=DEV)
        self.lists = torch.zeros((n, 512), dtype=torch.int32, device=DEV)
        self.counts = torch.zeros(n, dtype=torch.int32, device=DEV)
        for i in range(n):
            pick = torch.randperm(i + 1, generator=g, device=DEV)[:512].sort().values
            self.lists[i, :len(pick)] = pick.int()
            self.counts[i] = len(pick)

    def ring(self, kv: torch.Tensor, pos: int) -> torch.Tensor:
        ring = torch.zeros((128, kv.shape[1]), dtype=kv.dtype, device=DEV)
        for w in range(max(0, pos - 128), pos):
            ring[w % 128] = kv[w]
        return ring

    def run(self, pos: int, rows: int, prompt: bool, packed: bool, window_only: bool = False) -> torch.Tensor:
        kv, comp = (self.kv8, self.comp4) if packed else (self.kv, self.comp)
        out = torch.empty((rows, 32, 512), dtype=torch.bfloat16, device=DEV)
        part = torch.empty((rows, 32, ak.chunks(512), 514), device=DEV)
        lists = (None, None, None) if window_only else (comp, self.lists[pos:pos + rows], self.counts[pos:pos + rows])
        ak.attention(self.q[pos:pos + rows], self.ring(kv, pos), kv[pos:pos + rows],
                     torch.tensor([pos], dtype=torch.int32, device=DEV),
                     torch.arange(pos, pos + rows, dtype=torch.int32, device=DEV), *lists, self.sink, out,
                     prompt=prompt, part=part)
        return out


@pytest.mark.parametrize("window_only", [False, True])
@pytest.mark.parametrize("prompt", [False, True])
def test_attention_over_packed_caches_equals_bf16_caches(prompt, window_only):
    seq = Caches(1100, 3)
    for pos, rows in ((0, 6), (5, 3), (127, 6), (130, 1), (600, 6), (1090, 4)):
        assert torch.equal(seq.run(pos, rows, prompt, True, window_only),
                           seq.run(pos, rows, prompt, False, window_only)), (pos, rows)
    if prompt:
        assert torch.equal(seq.run(0, 1100, True, True, window_only), seq.run(0, 1100, True, False, window_only))


def test_dspark_block_over_a_packed_ring_equals_bf16():
    """Five rows anchored at p over the packed ring and the block's own bf16 rows."""

    seq = Caches(600, 4)
    p = 500
    bkv = seq.kv[p - 5:p].clone()
    pos = torch.tensor([p + 1], dtype=torch.int32, device=DEV)
    anchors = torch.full((5,), p, dtype=torch.int32, device=DEV)
    lists = torch.arange(5, dtype=torch.int32, device=DEV).repeat(5, 1)
    counts = torch.full((5,), 5, dtype=torch.int32, device=DEV)
    outs = []
    for kv in (seq.kv8, seq.kv):
        out = torch.empty((5, 32, 512), dtype=torch.bfloat16, device=DEV)
        part = torch.empty((5, 32, 3, 514), device=DEV)
        ak.attention(seq.q[:5], seq.ring(kv, p + 1), None, pos, anchors, bkv, lists, counts, seq.sink, out,
                     prompt=False, part=part)
        outs.append(out)
    assert torch.equal(outs[0], outs[1])


def test_attention_refuses_mixed_window_formats():
    seq = Caches(8, 5)
    out = torch.empty((1, 32, 512), dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError):
        ak.attention(seq.q[:1], seq.ring(seq.kv8, 1), seq.kv[1:2], torch.ones(1, dtype=torch.int32, device=DEV),
                     torch.ones(1, dtype=torch.int32, device=DEV), None, None, None, seq.sink, out, prompt=True)


@pytest.mark.parametrize("ratio,pos", [(1, 3000), (2, 5001), (1, 16381)])
def test_scores_over_packed_keys_equal_bf16_keys(ratio, pos):
    """Keys whose groups spread over 2^-6..2^6: packed keys give the bf16 keys' score bits."""

    g = torch.Generator(device=DEV).manual_seed(pos)
    n = (pos + MAX_ROWS) // ratio + 3
    q = quant.fp4_qdq_1x32_e8m0(torch.randn((MAX_ROWS, 32, 128), generator=g, device=DEV).to(torch.bfloat16))
    w = torch.randn((MAX_ROWS, 32), generator=g, device=DEV) * 0.05
    k = (torch.randn((n, 128), generator=g, device=DEV) * torch.rand((n, 1), generator=g, device=DEV) * 8)
    k = (k * 2.0 ** torch.randint(-6, 7, (n, 4), generator=g, device=DEV).repeat_interleave(32, 1)).to(torch.bfloat16)
    keys, keys4 = quant.fp4_qdq_1x32_e8m0(k.clone()), _packed(quant.FP4_E8M0, k)
    at = torch.tensor([pos], dtype=torch.int32, device=DEV)
    blocks = min(2048, n // 8)
    cand = torch.stack([torch.randperm(n // 8, generator=g, device=DEV)[:blocks].sort().values
                        for _ in range(MAX_ROWS)]).int()
    cand_n = torch.randint(1, blocks + 1, (MAX_ROWS,), generator=g, device=DEV).int()
    for kw in ({}, {"cand": cand, "cand_n": cand_n}):
        s = [ix.scores(q, w, kk, ratio, at, torch.full((MAX_ROWS, n), float("nan"), device=DEV), **kw)
             for kk in (keys4, keys)]
        assert torch.equal(s[0].view(torch.int32), s[1].view(torch.int32)), kw.keys()
