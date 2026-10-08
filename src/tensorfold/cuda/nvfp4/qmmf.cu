// Lane matmul for NVFP4 and FP8 weights (W4A16 / W8A16), exact weights in bf16 MMAs: per 16 (NVFP4), 32 (MXFP8) or 64
// inputs (FP8G: fp32 scales) acc = fma(P, block scale, acc), one tensor scale; K slices by shape, no row affects another.

#include <ATen/ATen.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "qmmf.cuh"

namespace {

using namespace qmmf_tile;

template <bool F32>
__global__ void reduce_kernel(const float* __restrict__ part, void* __restrict__ out, long long total, int SK,
                              float scale) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = part[i];
    for (int s = 1; s < SK; ++s) acc = acc + part[s * total + i];
    acc = acc * scale;
    if (F32) reinterpret_cast<float*>(out)[i] = acc;
    else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
}

template <int MODE, int BM, bool F32, bool CLUSTER, bool FUSE = false>
void launch(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs, double scale, at::Tensor& out,
            const at::Tensor& part, int N, int K, int SK, int npad) {
    constexpr int BN = 64, WM = 1, WN = 4, STAGES = 4;
    using T = Tile<MODE, BM, BN, WM, WN, STAGES>;
    const int M = x.size(0);
    auto kernel = qmmf_kernel<MODE, BM, BN, WM, WN, STAGES, F32, CLUSTER, FUSE>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        configured = true;
    }
    cudaLaunchConfig_t config = {};
    const int rows_t = (M + BM - 1) / BM;
    const int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / (static_cast<long long>(BM) * K * 2))));
    config.gridDim = dim3(rows_t * ((N + BN - 1) / BN), 1, FUSE ? 1 : SK);
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
        reinterpret_cast<const unsigned char*>(w.data_ptr()),
        bs.defined() ? reinterpret_cast<const uint8_t*>(bs.data_ptr()) : nullptr, static_cast<float>(scale),
        out.data_ptr(), part.defined() ? part.data_ptr<float>() : nullptr, M, N, K, SK, npad,
        M == 1 ? K : static_cast<int>(x.stride(0)), group));
}

template <int MODE, bool F32, bool CLUSTER>
void by_rows(int bm, const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs, double scale, at::Tensor& out,
             const at::Tensor& part, int N, int K, int SK, int npad) {
    switch (bm) {
        case 16: launch<MODE, 16, F32, CLUSTER>(x, w, bs, scale, out, part, N, K, SK, npad); break;
        case 32: launch<MODE, 32, F32, CLUSTER>(x, w, bs, scale, out, part, N, K, SK, npad); break;
        case 64: launch<MODE, 64, F32, CLUSTER>(x, w, bs, scale, out, part, N, K, SK, npad); break;
        default: launch<MODE, 64, F32, false, true>(x, w, bs, scale, out, part, N, K, SK, npad); break;   // 0: fused
    }
}

template <int MODE>
void by_output(int bm, bool f32, bool cluster, const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs,
               double scale, at::Tensor& out, const at::Tensor& part, int N, int K, int SK, int npad) {
    if (f32) { if (cluster) by_rows<MODE, true, true>(bm, x, w, bs, scale, out, part, N, K, SK, npad);
               else by_rows<MODE, true, false>(bm, x, w, bs, scale, out, part, N, K, SK, npad); }
    else { if (cluster) by_rows<MODE, false, true>(bm, x, w, bs, scale, out, part, N, K, SK, npad);
           else by_rows<MODE, false, false>(bm, x, w, bs, scale, out, part, N, K, SK, npad); }
}

}  // namespace

void qmmf_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs, double scale, at::Tensor& out,
               const at::Tensor& part, int64_t mode, int64_t N, int64_t K, int64_t SK, int64_t npad, int64_t bm,
               bool f32) {
    // slices add in one order via a cluster's shared memory (sm_90 on) or ``part`` and the reduce, or (bm 0, prompt
    // rows) in one block: the same bits
    const bool fused = bm == 0;
    const bool cluster = !fused && SK > 1 && SK <= 8 && !part.defined() &&
                         at::cuda::getCurrentDeviceProperties()->major >= 9;
    const int n = static_cast<int>(N), k = static_cast<int>(K), sk = static_cast<int>(SK), np = static_cast<int>(npad);
    at::Tensor slices = part;                         // sm_89: no clusters, so slices up to 8 meet here too
    if (!fused && SK > 1 && !cluster && !slices.defined())
        slices = at::empty({SK, x.size(0), N}, out.options().dtype(at::kFloat));
    const int b = static_cast<int>(bm);
    if (mode == FP4) by_output<FP4>(b, f32, cluster, x, w, bs, scale, out, slices, n, k, sk, np);
    else if (mode == FP8) by_output<FP8>(b, f32, cluster, x, w, bs, scale, out, slices, n, k, sk, np);
    else if (mode == MXFP8) by_output<MXFP8>(b, f32, cluster, x, w, bs, scale, out, slices, n, k, sk, np);
    else by_output<FP8G>(b, f32, cluster, x, w, bs, scale, out, slices, n, k, sk, np);
    if (!fused && SK > 1 && !cluster) {
        const long long total = static_cast<long long>(x.size(0)) * N;
        const int threads = 256, blocks = static_cast<int>((total + threads - 1) / threads);
        auto stream = at::cuda::getCurrentCUDAStream();
        if (f32) reduce_kernel<true><<<blocks, threads, 0, stream>>>(slices.data_ptr<float>(), out.data_ptr(), total,
                                                                     sk, static_cast<float>(scale));
        else reduce_kernel<false><<<blocks, threads, 0, stream>>>(slices.data_ptr<float>(), out.data_ptr(), total, sk,
                                                                  static_cast<float>(scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

namespace {

// NVFP4 weights staged for ``qmm_prefill8w`` (groups of 64): per (64-column tile, group) a block of 8 warps, a lane
// turning its two tiled words (16 codes of one column) into its 16 e4m3 bytes of the same fragment lane (the layouts
// share lanes), scaled by the column's max |value| / 448 over the group, rounded up to bf16.
__global__ void __launch_bounds__(256) stage_fp4_kernel(const uint32_t* __restrict__ words, const uint8_t* __restrict__ bs,
                                                        float global, uint8_t* __restrict__ w8,
                                                        __nv_bfloat16* __restrict__ scales, int KG, int npad) {
    constexpr int OFF[8] = {0, 8, 16, 24, 1, 9, 17, 25};  // nibble slot -> input offset (``qmm.OFFSETS``)
    constexpr int SLOT[8] = {0, 4, 1, 5, 2, 6, 3, 7};     // fragment byte -> nibble slot
    const int t = blockIdx.x, g = blockIdx.y, jj = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = t * 64 + jj * 8 + (lane >> 2), tig = lane & 3;
    const size_t at = ((static_cast<size_t>(t) * KG + g) * 8 + jj) * 32 + lane;
    float v[2][8];
    float top = 0.0f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t w = words[at * 2 + h];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int k = 32 * h + 2 * tig + OFF[p];
            const uint32_t code = (w >> (4 * p)) & 0xFu;
            const float mag = (code & 7u) < 2 ? 0.5f * (code & 1u) : static_cast<float>(1u << (((code & 7u) >> 1) - 1)) *
                                                                         (1.0f + 0.5f * (code & 1u));
            const float s = e4m3f(bs[((static_cast<size_t>(t) * KG + g) * 64 + jj * 8 + (lane >> 2)) * 4 + k / 16]);
            v[h][p] = ((code & 8u) ? -mag : mag) * s * global;
            top = fmaxf(top, fabsf(v[h][p]));
        }
    }
    top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
    top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
    const __nv_bfloat16 s = __float2bfloat16_ru(top > 0.0f ? top / 448.0f : 1.0f);
    if (tig == 0) scales[static_cast<size_t>(g) * npad + col] = s;
    const float sf = __bfloat162float(s);
    uint32_t out[2][2];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            uint32_t packed = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                packed |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(v[h][SLOT[r * 4 + i]] / sf, __NV_SATFINITE, __NV_E4M3))
                          << (8 * i);
            out[h][r] = packed;
        }
    uint2* dst = reinterpret_cast<uint2*>(w8 + (static_cast<size_t>(t) * KG + g) * 4096) + (jj * 32 + lane) * 2;
    dst[0] = make_uint2(out[0][0], out[0][1]);
    dst[1] = make_uint2(out[1][0], out[1][1]);
}

}  // namespace

void stage_fp4_cuda(const at::Tensor& words, const at::Tensor& bs, double global, at::Tensor& w8, at::Tensor& scales,
                    int64_t KG, int64_t npad) {
    dim3 grid(static_cast<unsigned>(npad / 64), static_cast<unsigned>(KG));
    stage_fp4_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(words.data_ptr()), reinterpret_cast<const uint8_t*>(bs.data_ptr()),
        static_cast<float>(global), reinterpret_cast<uint8_t*>(w8.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(scales.data_ptr()), static_cast<int>(KG), static_cast<int>(npad));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
