//============================================================================
// Silicon issue-rate probe: warp-level mma.sync loops on ZERO operands held in
// registers. No memory traffic and near-zero bit toggling, so the power governor
// never engages and the number approaches the tensor pipes' raw issue-rate bound
// FOR THE WARP-MMA (mma.sync) PATH.
//
// Read the result per architecture — the two are not the same measurement:
//   sm_120: warp-MMA is the full-rate tensor path, so this probe measures the
//     silicon bound.
//   sm_110: warp-MMA is a compatibility path (~0.5 inst/SM/clk here). The rated
//     tensor throughput lives on a path that the library conv kernels use and
//     public mma.sync cannot reach, so on this architecture the probe bounds
//     what mma.sync codegen can issue, NOT the silicon tensor bound. It is the
//     mechanism behind a low measured throughput on hand-written MatMul loops.
// Either way it never replaces the runtime-attainable ceiling in budgets.
//
// Known-truth accounting: ops are counted arithmetically, never measured —
//   ops = launches x blocks x warps/block x LOOP x UNROLL x ILP x ops/mma
// with ops/mma fixed by the instruction shape (m16n8k16 fp16 = 16*8*16*2 =
// 4096; m16n8k32 int8/fp8 = 8192; m16n8k8 tf32 = 2048): truth by construction.
//
// Build:  nvcc -O3 -arch=sm_110a peak_issue_probe.cu -o peak_issue_probe   (sm_110)
//         nvcc -O3 -arch=sm_120a peak_issue_probe.cu -o peak_issue_probe   (sm_120)
// Run:    ./peak_issue_probe [seconds_per_precision]   (default 1.0; use 3+
//         for a sustained pass — with zero toggling, sustained =~ burst is
//         itself a finding: it isolates the DATA-dependence of throttling)
//============================================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define LOOP   2048        // outer loop inside kernel (runtime-fixed)
#define UNROLL 32          // unrolled mma per loop iteration per accumulator
#define ILP    8           // independent accumulator sets (latency cover; 8 fills the f16-acc window)

// Every warp executes LOOP*UNROLL*ILP mma instructions per kernel launch.
static const long long MMA_PER_WARP = (long long)LOOP * UNROLL * ILP;

//---------------------------------------------------------------------------
// fp16 x fp16 -> fp16 accum, m16n8k16 (max-rate half path): 4096 ops/mma
__global__ void k_fp16(unsigned *sink, const unsigned *z) {
    int t = threadIdx.x & 31;
    unsigned a0=z[t],a1=z[t+32],a2=z[t+64],a3=z[t+96],b0=z[t+128],b1=z[t+160];
    unsigned c0[ILP], c1[ILP];
    for (int i = 0; i < ILP; ++i) { c0[i] = z[t+i]; c1[i] = z[t+i+8]; }
    for (int l = 0; l < LOOP; ++l) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int i = 0; i < ILP; ++i) {
                asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                  "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                  : "+r"(c0[i]), "+r"(c1[i])
                  : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }
    unsigned acc = 0;
    for (int i = 0; i < ILP; ++i) acc ^= c0[i] ^ c1[i];
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = acc;
}

//---------------------------------------------------------------------------
// bf16 x bf16 -> fp32 accum, m16n8k16: 4096 ops/mma
__global__ void k_bf16(unsigned *sink, const unsigned *z) {
    int t = threadIdx.x & 31;
    unsigned a0=z[t],a1=z[t+32],a2=z[t+64],a3=z[t+96],b0=z[t+128],b1=z[t+160];
    float c[ILP][4];
    const float *zf = (const float*)z;
    for (int i = 0; i < ILP; ++i) { c[i][0]=zf[t+i]; c[i][1]=zf[t+i+8]; c[i][2]=zf[t+i+16]; c[i][3]=zf[t+i+24]; }
    for (int l = 0; l < LOOP; ++l) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int i = 0; i < ILP; ++i) {
                asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                  : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                  : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }
    float accf = 0;
    for (int i = 0; i < ILP; ++i) accf += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = (unsigned)accf;
}

//---------------------------------------------------------------------------
// tf32 x tf32 -> fp32 accum, m16n8k8: 2048 ops/mma
__global__ void k_tf32(unsigned *sink, const unsigned *z) {
    int t = threadIdx.x & 31;
    unsigned a0=z[t],a1=z[t+32],a2=z[t+64],a3=z[t+96],b0=z[t+128],b1=z[t+160];
    float c[ILP][4];
    const float *zf = (const float*)z;
    for (int i = 0; i < ILP; ++i) { c[i][0]=zf[t+i]; c[i][1]=zf[t+i+8]; c[i][2]=zf[t+i+16]; c[i][3]=zf[t+i+24]; }
    for (int l = 0; l < LOOP; ++l) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int i = 0; i < ILP; ++i) {
                asm volatile(
                  "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                  : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                  : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }
    float accf = 0;
    for (int i = 0; i < ILP; ++i) accf += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = (unsigned)accf;
}

//---------------------------------------------------------------------------
// int8 x int8 -> s32 accum, m16n8k32: 8192 ops/mma
__global__ void k_int8(unsigned *sink, const unsigned *z) {
    int t = threadIdx.x & 31;
    unsigned a0=z[t],a1=z[t+32],a2=z[t+64],a3=z[t+96],b0=z[t+128],b1=z[t+160];
    int c[ILP][4];
    const int *zi = (const int*)z;
    for (int i = 0; i < ILP; ++i) { c[i][0]=zi[t+i]; c[i][1]=zi[t+i+8]; c[i][2]=zi[t+i+16]; c[i][3]=zi[t+i+24]; }
    for (int l = 0; l < LOOP; ++l) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int i = 0; i < ILP; ++i) {
                asm volatile(
                  "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                  : "+r"(c[i][0]), "+r"(c[i][1]), "+r"(c[i][2]), "+r"(c[i][3])
                  : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }
    float accf = 0;
    for (int i = 0; i < ILP; ++i) accf += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = (unsigned)accf;
}

#ifndef NO_FP8
//---------------------------------------------------------------------------
// fp8(e4m3) x fp8 -> fp32 accum, m16n8k32: 8192 ops/mma
__global__ void k_fp8(unsigned *sink, const unsigned *z) {
    int t = threadIdx.x & 31;
    unsigned a0=z[t],a1=z[t+32],a2=z[t+64],a3=z[t+96],b0=z[t+128],b1=z[t+160];
    float c[ILP][4];
    const float *zf = (const float*)z;
    for (int i = 0; i < ILP; ++i) { c[i][0]=zf[t+i]; c[i][1]=zf[t+i+8]; c[i][2]=zf[t+i+16]; c[i][3]=zf[t+i+24]; }
    for (int l = 0; l < LOOP; ++l) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int i = 0; i < ILP; ++i) {
                asm volatile(
                  "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                  : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                  : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }
    float accf = 0;
    for (int i = 0; i < ILP; ++i) accf += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = (unsigned)accf;
}
#endif

//---------------------------------------------------------------------------
struct Probe { const char *name; void (*kern)(unsigned*, const unsigned*); double ops_per_mma; };

int main(int argc, char **argv) {
    double target_s = (argc > 1) ? atof(argv[1]) : 1.0;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sms = prop.multiProcessorCount;
    int blocks = sms * 2;                 // 2 CTAs/SM
    int threads = 256;                    // 8 warps/CTA
    long long warps = (long long)blocks * (threads / 32);

    printf("device: %s  sm_%d%d  SMs=%d  blocks=%d threads=%d warps=%lld\n",
           prop.name, prop.major, prop.minor, sms, blocks, threads, warps);
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

    unsigned *sink, *zbuf;
    cudaMalloc(&sink, sizeof(unsigned));
    cudaMalloc(&zbuf, 4096);
    cudaMemset(zbuf, 0, 4096);           // runtime zeros — opaque to ptxas

    for (auto &p : probes) {
        if (p.name[0] == 'f' && p.name[1] == 'p' && p.name[2] == '8' && prop.major == 11) {
            printf("%-26s SKIPPED on sm_110: public ptxas emits no fp8 warp-MMA "
                   "(SASS-verified) — int8 is the same-rung proxy; TRT attainable 445.9 "
                   "remains the measured fp8 evidence\n", p.name);
            continue;
        }
        // warmup + single-launch timing to size the run
        p.kern<<<blocks, threads>>>(sink, zbuf);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            printf("%-26s UNSUPPORTED on this arch (%s)\n", p.name,
                   cudaGetErrorString(cudaGetLastError()));
            cudaGetLastError();
            continue;
        }
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        p.kern<<<blocks, threads>>>(sink, zbuf);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float one_ms = 0; cudaEventElapsedTime(&one_ms, t0, t1);
        int launches = (int)(target_s * 1000.0 / one_ms) + 1;

        cudaEventRecord(t0);
        for (int i = 0; i < launches; ++i) p.kern<<<blocks, threads>>>(sink, zbuf);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0; cudaEventElapsedTime(&ms, t0, t1);

        double ops = (double)launches * warps * MMA_PER_WARP * p.ops_per_mma;
        double tops = ops / (ms * 1e-3) / 1e12;
        printf("%-26s %8.1f T%s   (%d launches, %.0f ms)\n",
               p.name, tops, (p.ops_per_mma > 4000.0 && p.name[0]=='i') ? "OPS" : "FLOPS",
               launches, ms);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    cudaFree(sink); cudaFree(zbuf);
    return 0;
}
