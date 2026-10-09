// Paced L2 warming: a 16-byte asynchronous copy from every 32-byte sector of a table of byte ranges, in table order,
// at a capped rate, into shared scratch nobody reads, so the kernels after it find those sectors in L2.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int THREADS = 128;
constexpr int64_t SECTOR = 32;                  // bytes one copy brings into L2 (it reads 16 of them)
constexpr int STAGES = 8;                       // chunks a block keeps in flight
constexpr int MAX_RANGES = 8;

__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// A copy, not a load: ptxas drops a load whose value nobody reads. .cg caches the sector in L2, not L1.
__device__ __forceinline__ void fetch(void* dst, const char* src) {
    const uint32_t to = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(to), "l"(src));
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// table: int64 [n, 2] of (sector-aligned address, sectors). Block b reads chunks b, b + grid, ... of THREADS sectors
// in table order, its j-th issued no earlier than j * ns_per_chunk after it started.
__global__ void __launch_bounds__(THREADS) warm_kernel(const int64_t* __restrict__ table, int n, double ns_per_chunk) {
    __shared__ int64_t base[MAX_RANGES], end[MAX_RANGES + 1];
    __shared__ uint64_t t0;
    __shared__ __align__(16) uint4 sink[STAGES][THREADS];
    if (threadIdx.x == 0) {
        end[0] = 0;
        for (int i = 0; i < n; ++i) {
            base[i] = table[2 * i];
            end[i + 1] = end[i] + table[2 * i + 1];
        }
        t0 = now_ns();
    }
    __syncthreads();
    const int64_t total = end[n];
    int r = 0;
    int64_t issued = 0;
    for (int64_t c = blockIdx.x; c * THREADS < total; c += gridDim.x) {
        const int64_t v = c * THREADS + threadIdx.x;
        wait<STAGES - 1>();                     // the chunk STAGES back has landed: its slot is free
        if (v < total) {
            while (v >= end[r + 1]) ++r;
            fetch(&sink[issued % STAGES][threadIdx.x], reinterpret_cast<const char*>(base[r]) + (v - end[r]) * SECTOR);
        }
        commit();
        ++issued;
        if (threadIdx.x == 0) {
            const uint64_t due = t0 + static_cast<uint64_t>(static_cast<double>(issued) * ns_per_chunk);
            while (now_ns() < due) __nanosleep(128);
        }
        __syncthreads();
    }
    wait<0>();
}

}  // namespace

void warm_cuda(const at::Tensor& table, double gb_per_s, int64_t ctas) {
    const int n = static_cast<int>(table.size(0));
    if (n == 0) return;
    // the grid together at gb_per_s (bytes a nanosecond): each block its share, a chunk at a time
    const double ns_per_chunk = static_cast<double>(THREADS * SECTOR) * static_cast<double>(ctas) / gb_per_s;
    warm_kernel<<<static_cast<unsigned>(ctas), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        table.data_ptr<int64_t>(), n, ns_per_chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
