//============================================================================
// Silicon issue-rate probe: warp-level mma.sync loops on ZERO operands held in
// registers. No memory traffic and near-zero bit toggling, so the power governor
// never engages and the number approaches the tensor pipes' raw issue-rate bound
// for the warp-MMA (mma.sync) path. It never replaces the runtime-attainable
// ceiling in budgets: on sm_120 warp-MMA is the full-rate tensor path (silicon
// bound); on sm_110 it is a ~0.5 inst/SM/clk compatibility path, so the probe
// bounds what mma.sync codegen can issue and explains a low hand-written MatMul.
//
// Ops are counted arithmetically, never measured (truth by construction):
//   ops = launches x blocks x warps/block x LOOP x UNROLL x ILP x ops/mma
// with ops/mma fixed by the instruction shape (m16n8k16=4096, m16n8k32=8192,
// m16n8k8=2048).
//
// Build:  nvcc -O3 -arch=sm_110a peak_issue_probe.cu -o peak_issue_probe  (or sm_120a)
// Run:    ./peak_issue_probe [seconds_per_precision]   (default 1.0; 3+ for a
//         sustained pass — with zero toggling, sustained =~ burst isolates the
//         DATA-dependence of throttling)
// Output: one "device:" line, two header lines, then per precision
//         "<name> <T-rate> T{FLOPS|OPS}   (<launches> launches, <ms> ms)".
//============================================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define LOOP   2048        // outer loop inside kernel (runtime-fixed)
#define UNROLL 32          // unrolled mma per loop iteration per accumulator
#define ILP    8           // independent accumulator sets (8 fills the f16-acc latency window)

static const long long MMA_PER_WARP = (long long)LOOP * UNROLL * ILP;

// A/B fragments come from a runtime-zeroed buffer so ptxas cannot fold them.
struct Fragments { unsigned a0, a1, a2, a3, b0, b1; };

__device__ __forceinline__ Fragments load_fragments(const unsigned *zeros, int lane) {
    Fragments f;
    f.a0 = zeros[lane];        f.a1 = zeros[lane + 32];  f.a2 = zeros[lane + 64];
    f.a3 = zeros[lane + 96];   f.b0 = zeros[lane + 128]; f.b1 = zeros[lane + 160];
    return f;
}

template <typename T>
__device__ __forceinline__ void init_accumulators(T c[ILP][4], const T *zeros, int lane) {
    for (int i = 0; i < ILP; ++i) {
        c[i][0] = zeros[lane + i];      c[i][1] = zeros[lane + i + 8];
        c[i][2] = zeros[lane + i + 16]; c[i][3] = zeros[lane + i + 24];
    }
}

// fp16 x fp16 -> fp16 accum, m16n8k16 (max-rate half path): 4096 ops/mma
__global__ void k_fp16(unsigned *sink, const unsigned *zeros) {
    int lane = threadIdx.x & 31;
    Fragments f = load_fragments(zeros, lane);
    unsigned c0[ILP], c1[ILP];
    for (int i = 0; i < ILP; ++i) { c0[i] = zeros[lane + i]; c1[i] = zeros[lane + i + 8]; }
    for (int iteration = 0; iteration < LOOP; ++iteration) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int i = 0; i < ILP; ++i) {
                asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                  "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                  : "+r"(c0[i]), "+r"(c1[i])
                  : "r"(f.a0), "r"(f.a1), "r"(f.a2), "r"(f.a3), "r"(f.b0), "r"(f.b1));
            }
        }
    }
    unsigned acc = 0;
    for (int i = 0; i < ILP; ++i) acc ^= c0[i] ^ c1[i];
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = acc;
}

// bf16 x bf16 -> fp32 accum, m16n8k16: 4096 ops/mma
__global__ void k_bf16(unsigned *sink, const unsigned *zeros) {
    int lane = threadIdx.x & 31;
    Fragments f = load_fragments(zeros, lane);
    float c[ILP][4];
    init_accumulators(c, (const float *)zeros, lane);
    for (int iteration = 0; iteration < LOOP; ++iteration) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int i = 0; i < ILP; ++i) {
                asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                  : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                  : "r"(f.a0), "r"(f.a1), "r"(f.a2), "r"(f.a3), "r"(f.b0), "r"(f.b1));
            }
        }
    }
    float sum = 0;
    for (int i = 0; i < ILP; ++i) sum += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = (unsigned)sum;
}

// tf32 x tf32 -> fp32 accum, m16n8k8: 2048 ops/mma
__global__ void k_tf32(unsigned *sink, const unsigned *zeros) {
    int lane = threadIdx.x & 31;
    Fragments f = load_fragments(zeros, lane);
    float c[ILP][4];
    init_accumulators(c, (const float *)zeros, lane);
    for (int iteration = 0; iteration < LOOP; ++iteration) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int i = 0; i < ILP; ++i) {
                asm volatile(
                  "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                  : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                  : "r"(f.a0), "r"(f.a1), "r"(f.a2), "r"(f.a3), "r"(f.b0), "r"(f.b1));
            }
        }
    }
    float sum = 0;
    for (int i = 0; i < ILP; ++i) sum += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = (unsigned)sum;
}

// int8 x int8 -> s32 accum, m16n8k32: 8192 ops/mma
__global__ void k_int8(unsigned *sink, const unsigned *zeros) {
    int lane = threadIdx.x & 31;
    Fragments f = load_fragments(zeros, lane);
    int c[ILP][4];
    init_accumulators(c, (const int *)zeros, lane);
    for (int iteration = 0; iteration < LOOP; ++iteration) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int i = 0; i < ILP; ++i) {
                asm volatile(
                  "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                  : "+r"(c[i][0]), "+r"(c[i][1]), "+r"(c[i][2]), "+r"(c[i][3])
                  : "r"(f.a0), "r"(f.a1), "r"(f.a2), "r"(f.a3), "r"(f.b0), "r"(f.b1));
            }
        }
    }
    float sum = 0;
    for (int i = 0; i < ILP; ++i) sum += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = (unsigned)sum;
}

#ifndef NO_FP8
// fp8(e4m3) x fp8 -> fp32 accum, m16n8k32: 8192 ops/mma
__global__ void k_fp8(unsigned *sink, const unsigned *zeros) {
    int lane = threadIdx.x & 31;
    Fragments f = load_fragments(zeros, lane);
    float c[ILP][4];
    init_accumulators(c, (const float *)zeros, lane);
    for (int iteration = 0; iteration < LOOP; ++iteration) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int i = 0; i < ILP; ++i) {
                asm volatile(
                  "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                  : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                  : "r"(f.a0), "r"(f.a1), "r"(f.a2), "r"(f.a3), "r"(f.b0), "r"(f.b1));
            }
        }
    }
    float sum = 0;
    for (int i = 0; i < ILP; ++i) sum += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = (unsigned)sum;
}
#endif

struct Probe { const char *name; void (*kern)(unsigned*, const unsigned*); double ops_per_mma; };

static bool is_fp8(const Probe &probe) {
    return probe.name[0] == 'f' && probe.name[1] == 'p' && probe.name[2] == '8';
}

static const char *rate_unit(const Probe &probe) {
    return (probe.ops_per_mma > 4000.0 && probe.name[0] == 'i') ? "OPS" : "FLOPS";
}

// Times `launches` back-to-back launches of one probe; returns elapsed ms.
static float time_launches(const Probe &probe, int launches, int blocks, int threads,
                           unsigned *sink, unsigned *zeros) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < launches; ++i) probe.kern<<<blocks, threads>>>(sink, zeros);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

int main(int argc, char **argv) {
    double target_s = (argc > 1) ? atof(argv[1]) : 1.0;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;
    int blocks = sm_count * 2;            // 2 CTAs/SM
    int threads = 256;                    // 8 warps/CTA
    long long warps = (long long)blocks * (threads / 32);

    printf("device: %s  sm_%d%d  SMs=%d  blocks=%d threads=%d warps=%lld\n",
           prop.name, prop.major, prop.minor, sm_count, blocks, threads, warps);
    printf("mma per warp per launch: %lld  (LOOP=%d UNROLL=%d ILP=%d)\n",
           MMA_PER_WARP, LOOP, UNROLL, ILP);
    printf("target per precision: %.1f s   (ops counted arithmetically)\n\n", target_s);

    Probe probes[] = {
        {"fp16 (f16 acc, m16n8k16)", k_fp16, 4096.0},
        {"bf16 (f32 acc, m16n8k16)", k_bf16, 4096.0},
        {"tf32 (f32 acc, m16n8k8) ", k_tf32, 2048.0},
        {"int8 (s32 acc, m16n8k32)", k_int8, 8192.0},
#ifndef NO_FP8
        {"fp8  (f32 acc, m16n8k32)", k_fp8,  8192.0},
#endif
    };

    unsigned *sink, *zeros;
    cudaMalloc(&sink, sizeof(unsigned));
    cudaMalloc(&zeros, 4096);
    cudaMemset(zeros, 0, 4096);          // runtime zeros — opaque to ptxas

    for (auto &probe : probes) {
        // Public ptxas emits no fp8 warp-MMA on sm_110 (SASS-verified).
        if (is_fp8(probe) && prop.major == 11) {
            printf("%-26s SKIPPED on sm_110: public ptxas emits no fp8 warp-MMA "
                   "(SASS-verified) — int8 is the same-rung proxy; TRT attainable 445.9 "
                   "remains the measured fp8 evidence\n", probe.name);
            continue;
        }
        probe.kern<<<blocks, threads>>>(sink, zeros);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            printf("%-26s UNSUPPORTED on this arch (%s)\n", probe.name,
                   cudaGetErrorString(cudaGetLastError()));
            cudaGetLastError();
            continue;
        }
        // One timed launch sizes the run to the requested wall time.
        float one_ms = time_launches(probe, 1, blocks, threads, sink, zeros);
        int launches = (int)(target_s * 1000.0 / one_ms) + 1;
        float ms = time_launches(probe, launches, blocks, threads, sink, zeros);

        double ops = (double)launches * warps * MMA_PER_WARP * probe.ops_per_mma;
        double tera_rate = ops / (ms * 1e-3) / 1e12;
        printf("%-26s %8.1f T%s   (%d launches, %.0f ms)\n",
               probe.name, tera_rate, rate_unit(probe), launches, ms);
    }
    cudaFree(sink); cudaFree(zeros);
    return 0;
}
