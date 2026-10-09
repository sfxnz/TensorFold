#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

void warm_cuda(const at::Tensor&, double, int64_t);

// Warm the byte ranges of ``table`` (int64 [n, 2]: 32-byte-aligned address, 32-byte sectors) into L2 on the current
// stream, ``ctas`` blocks reading together at ``gb_per_s``.
void warm(const at::Tensor& table, double gb_per_s, int64_t ctas) {
    TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kLong && table.dim() == 2 && table.size(1) == 2 &&
                table.is_contiguous() && table.size(0) <= 8, "table: int64 [n <= 8, 2] on the GPU");
    TORCH_CHECK(std::isfinite(gb_per_s) && gb_per_s > 0 && ctas >= 1 && ctas <= 64, "rate > 0, 1 to 64 blocks");
    c10::cuda::CUDAGuard guard(table.device());
    warm_cuda(table, gb_per_s, ctas);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warm", &warm);
}
