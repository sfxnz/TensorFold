// DeepSeek's MXFP8 decode projections on qmmf's kernel with tiles chosen by shape: columns a block, pipeline stages,
// and K slices meeting in a cluster or in one block. Each output keeps its slices, group order and slice order.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "qmmf.cuh"

namespace {

using namespace qmmf_tile;

template <int BM, int BN, int STAGES, bool F32, bool CLUSTER, bool FUSE>
void launch(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs, at::Tensor& out, int N, int K, int SK,
            int npad) {
    constexpr int WM = 1, WN = BN / 16;                    // two n8 tiles a warp
    using T = Tile<MXFP8, BM, BN, WM, WN, STAGES>;
    const int M = x.size(0);
    auto kernel = qmmf_kernel<MXFP8, BM, BN, WM, WN, STAGES, F32, CLUSTER, FUSE>;
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
        configured = true;
    }
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3((N + BN - 1) / BN, 1, FUSE ? 1 : SK);   // one row tile holds every row
    config.blockDim = dim3(T::THREADS);
    config.dynamicSmemBytes = T::SMEM;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    if (CLUSTER) {
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 1;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = SK;
        config.attrs = attr;
        config.numAttrs = 1;
    }
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const unsigned char*>(w.data_ptr()), reinterpret_cast<const uint8_t*>(bs.data_ptr()), 1.0f,
        out.data_ptr(), static_cast<float*>(nullptr), M, N, K, SK, npad,
        M == 1 ? K : static_cast<int>(x.stride(0)), 1));
}

template <int BM, int BN, int STAGES>
void by_reduce(bool fuse, bool f32, const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs, at::Tensor& out,
               int N, int K, int SK, int npad) {
    if (fuse) { if (f32) launch<BM, BN, STAGES, true, false, true>(x, w, bs, out, N, K, SK, npad);
                else launch<BM, BN, STAGES, false, false, true>(x, w, bs, out, N, K, SK, npad); }
    else { if (f32) launch<BM, BN, STAGES, true, true, false>(x, w, bs, out, N, K, SK, npad);
           else launch<BM, BN, STAGES, false, true, false>(x, w, bs, out, N, K, SK, npad); }
}

template <int BM, int BN>
void by_stages(int stages, bool fuse, bool f32, const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs,
               at::Tensor& out, int N, int K, int SK, int npad) {
    switch (stages) {
        case 4: by_reduce<BM, BN, 4>(fuse, f32, x, w, bs, out, N, K, SK, npad); break;
        case 6: by_reduce<BM, BN, 6>(fuse, f32, x, w, bs, out, N, K, SK, npad); break;
        default: by_reduce<BM, BN, 8>(fuse, f32, x, w, bs, out, N, K, SK, npad); break;
    }
}

template <int BM>
void by_columns(int bn, int stages, bool fuse, bool f32, const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs,
                at::Tensor& out, int N, int K, int SK, int npad) {
    switch (bn) {
        case 16: by_stages<BM, 16>(stages, fuse, f32, x, w, bs, out, N, K, SK, npad); break;
        case 32: by_stages<BM, 32>(stages, fuse, f32, x, w, bs, out, N, K, SK, npad); break;
        default: by_stages<BM, 64>(stages, fuse, f32, x, w, bs, out, N, K, SK, npad); break;
    }
}

}  // namespace

void gemv_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs, at::Tensor& out, int64_t N, int64_t SK,
               int64_t npad, int64_t bn, int64_t stages, bool fuse, bool f32) {
    // one slice needs no meeting; before sm_90 (no clusters) slices meet in the block: the cluster's order either way
    const bool in_block = fuse || SK == 1 || at::cuda::getCurrentDeviceProperties()->major < 9;
    const int n = static_cast<int>(N), k = static_cast<int>(x.size(1)), sk = static_cast<int>(SK);
    const int np = static_cast<int>(npad), b = static_cast<int>(bn), st = static_cast<int>(stages);
    if (x.size(0) <= 16) by_columns<16>(b, st, in_block, f32, x, w, bs, out, n, k, sk, np);
    else by_columns<32>(b, st, in_block, f32, x, w, bs, out, n, k, sk, np);
}
