#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void gemv_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t,
               int64_t, bool, bool);

// out (M, n) = x (M, K) bf16 @ an MXFP8 projection's bytes, M 1 to 32: ``bn`` columns a block (16, 32 or 64),
// ``stages`` (4, 6 or 8), the ``sk`` K slices summed in one block (``fuse``) or a cluster; fp32 or bf16 out.
void gemv(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs, at::Tensor out, int64_t n, int64_t sk,
          int64_t npad, int64_t bn, int64_t stages, bool fuse, bool f32) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1, "x: (M, K) bf16");
    const int64_t m = x.size(0), k = x.size(1);
    TORCH_CHECK(m >= 1 && m <= 32, "decode rows: 1 to 32");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (m == 1 || x.stride(0) % 8 == 0),
                "x rows on 16 bytes");
    TORCH_CHECK(sk >= 1 && sk <= 8 && k % 64 == 0 && (k / 64) % sk == 0, "K in whole groups of 64, 1 to 8 slices");
    TORCH_CHECK(npad % 64 == 0 && n <= npad, "n within npad, npad in 64-column tiles");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.numel() * w.element_size() == npad * k,
                "weight bytes do not match npad and K");
    TORCH_CHECK(bs.is_cuda() && bs.is_contiguous() && bs.numel() == (k / 64) * npad * 2,
                "block scales [npad/64, K/64, 64, 2]");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    TORCH_CHECK(bn == 16 || bn == 32 || bn == 64, "bn: 16, 32 or 64 columns");
    TORCH_CHECK(stages == 4 || stages == 6 || stages == 8, "stages: 4, 6 or 8");
    c10::cuda::CUDAGuard guard(x.device());
    gemv_cuda(x, w, bs, out, n, sk, npad, bn, stages, fuse, f32);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv", &gemv);
}
