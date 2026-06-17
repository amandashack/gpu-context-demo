"""PROTOTYPE — throwaway. Validates the Phase 2 "tells vs traps" kernel gallery.

DELETE ME once the verdict is captured (see prototype_regime_tells_NOTES.md).

Question this answers
---------------------
For each exhibit kernel we predict a regime from the *source* (static arithmetic
intensity vs the A100 roofline ridge, plus compile stats) and then *measure* the
real operating point (achieved GFLOP/s and apparent GB/s from GPU timing). The
point is to see which predictions hold and which are TRAPS — cases where the
source-level guess is wrong and you can only know by measuring:

  - L2 cache reuse   -> source says HBM-bound, but traffic is served from L2
  - register spills  -> source looks compute-bound, but spills add memory traffic
  - uncoalesced acc. -> same source shape as a clean kernel, but effective BW collapses

Run on the SLAC Ampere node (needs a real CUDA GPU + cupy):
    uv run python prototype_regime_tells.py

No persistence, no abstractions, verbose on purpose.
"""

import cupy as cp
from cupyx.profiler import benchmark

# --- device peaks (detected; compute peak uses a CC->FP32-cores/SM table) -----
props = cp.cuda.runtime.getDeviceProperties(0)
name = props["name"].decode()
cc = (props["major"], props["minor"])
sm = props["multiProcessorCount"]
clock_khz = props["clockRate"]
mem_khz = props["memoryClockRate"]
bus_bits = props["memoryBusWidth"]

FP32_CORES_PER_SM = {(7, 0): 64, (7, 2): 64, (7, 5): 64,
                     (8, 0): 64, (8, 6): 128, (8, 9): 128, (9, 0): 128}
cores = FP32_CORES_PER_SM.get(cc, 64)  # A100 (8.0) -> 64

peak_fp32_gflops = sm * cores * 2 * (clock_khz * 1e3) / 1e9
peak_hbm_gbps = 2 * (mem_khz * 1e3) * (bus_bits / 8) / 1e9
ridge = peak_fp32_gflops / peak_hbm_gbps  # FLOPs/byte at the knee

print(f"GPU: {name}  CC {cc[0]}.{cc[1]}  {sm} SMs")
print(f"peak FP32 ~ {peak_fp32_gflops/1e3:.1f} TFLOP/s  ({cores} FP32 cores/SM assumed)")
print(f"peak HBM  ~ {peak_hbm_gbps:.0f} GB/s   ridge ~ {ridge:.1f} FLOPs/byte")
print("=" * 78)


def measure(kernel, grid, block, args, n_repeat=50, n_warmup=10):
    """GPU-time a kernel; return mean seconds. Touch attrs to force the compile."""
    _ = kernel.num_regs  # forces NVRTC compile so the stats below are populated
    r = benchmark(lambda: kernel(grid, block, args), n_repeat=n_repeat, n_warmup=n_warmup)
    return r.gpu_times.mean()


def report(label, src_ai, src_bytes, flops, t_s, kernel, predicted, note=""):
    apparent_gbps = src_bytes / t_s / 1e9   # uses the *source-counted* bytes
    gflops = flops / t_s / 1e9
    frac_bw = apparent_gbps / peak_hbm_gbps
    frac_fl = gflops / peak_fp32_gflops
    spill = kernel.local_size_bytes

    if apparent_gbps > peak_hbm_gbps * 1.1:
        measured = "cache-served (NOT DRAM-bound)"  # >100% of HBM is physically impossible
    elif frac_bw >= frac_fl:
        measured = "HBM-bound"
    else:
        measured = "compute-bound"

    trap = predicted.split("-")[0] not in measured
    print(f"\n[{label}]  {note}")
    print(f"  source AI ~ {src_ai:6.2f} FLOPs/byte  -> predicted: {predicted}")
    print(f"  compile:   regs/thread={kernel.num_regs}  smem={kernel.shared_size_bytes}B  "
          f"spill(local)={spill}B")
    print(f"  measured:  {gflops:8.0f} GFLOP/s ({frac_fl*100:4.1f}% peak)   "
          f"apparent {apparent_gbps:7.0f} GB/s ({frac_bw*100:5.1f}% HBM)")
    print(f"  VERDICT:   {measured}" + ("   <-- TRAP (source guess was wrong)" if trap else ""))


N = 1 << 24  # 16M floats = 64 MB; exceeds the 40 MB L2 so streaming is real DRAM
BLOCK = 256
GRID = (N + BLOCK - 1) // BLOCK

# === Exhibit 1: AI-dial — intuition WORKS (sweep across the ridge) ============
ai_src = r"""
extern "C" __global__ void ai_dial(const float* in, float* out, int n, int iters) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        #pragma unroll 1
        for (int k = 0; k < iters; k++) x = fmaf(x, 1.0001f, 1e-7f);
        out[i] = x;
    }
}
"""
ai_kernel = cp.RawKernel(ai_src, "ai_dial")
a_in = cp.ones(N, dtype=cp.float32)
a_out = cp.empty(N, dtype=cp.float32)
print("\n### Exhibit 1: AI-dial (control — prediction should hold at both ends)")
for iters in (1, 8, 64, 512):
    t = measure(ai_kernel, (GRID,), (BLOCK,), (a_in, a_out, N, iters))
    bytes_ = 8 * N          # read + write one float each
    flops = 2 * N * iters   # one FMA = 2 FLOPs
    ai = flops / bytes_
    pred = "compute-bound" if ai > ridge else "HBM-bound"
    report("ai_dial", ai, bytes_, flops, t, ai_kernel, pred, note=f"iters={iters}")

# === Exhibit 2: SAXPY — textbook HBM-bound, intuition WORKS ===================
saxpy_src = r"""
extern "C" __global__ void saxpy(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) c[i] = 2.0f*a[i] + b[i];
}
"""
saxpy_kernel = cp.RawKernel(saxpy_src, "saxpy")
b = cp.ones(N, dtype=cp.float32)
c = cp.empty(N, dtype=cp.float32)
print("\n### Exhibit 2: SAXPY (control — should read HBM-bound)")
t = measure(saxpy_kernel, (GRID,), (BLOCK,), (a_in, b, c, N))
report("saxpy", 2 / 12, 12 * N, 2 * N, t, saxpy_kernel, "HBM-bound")

# === Exhibit 3: L2-reuse TRAP — small buffer reread, source says HBM ==========
l2_src = r"""
extern "C" __global__ void l2_reuse(const float* small, float* out, int m, int reuse) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float acc = 0.0f;
    #pragma unroll 1
    for (int k = 0; k < reuse; k++) acc += small[(i + k) % m];   // re-reads a small buffer
    out[i % m] = acc;
}
"""
l2_kernel = cp.RawKernel(l2_src, "l2_reuse")
M = 1 << 20  # 4 MB — fits comfortably in the 40 MB L2
REUSE = 64
small = cp.ones(M, dtype=cp.float32)
l2_out = cp.empty(M, dtype=cp.float32)
print("\n### Exhibit 3: L2-reuse (TRAP — source counts huge traffic, L2 serves it)")
t = measure(l2_kernel, (GRID,), (BLOCK,), (small, l2_out, M, REUSE))
src_bytes = 4 * N * REUSE     # every access counted as a DRAM read
report("l2_reuse", (N * REUSE) / src_bytes, src_bytes, N * REUSE, t, l2_kernel, "HBM-bound")

# === Exhibit 4: register-spill TRAP — same source, capped registers ===========
reg_src = r"""
extern "C" __global__ void reg_heavy(const float* in, float* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float r[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) r[j] = in[i] * (1.0f + 0.01f*j);
    #pragma unroll 1
    for (int k = 0; k < 64; k++)
        #pragma unroll
        for (int j = 0; j < 16; j++) r[j] = fmaf(r[j], r[(j+1)&15], r[(j+7)&15]);
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; j++) s += r[j];
    out[i] = s;
}
"""
# Same body as reg_heavy, but the inner index into r[] is RUNTIME-dependent (d). A
# register file isn't addressable, so any array indexed by a runtime value is forced
# into local memory -> local_size_bytes > 0. This is the reliable, toolchain-proof way
# to show hidden memory traffic the source doesn't advertise (register caps via
# --maxrregcount / __launch_bounds__ did NOT bind through CuPy's JIT path).
reg_src_local = r"""
extern "C" __global__ void reg_heavy_local(const float* in, float* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float r[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) r[j] = in[i] * (1.0f + 0.01f*j);
    #pragma unroll 1
    for (int k = 0; k < 64; k++) {
        int d = k & 15;   // runtime index -> r[] can't live in registers -> local memory
        #pragma unroll
        for (int j = 0; j < 16; j++) r[j] = fmaf(r[j], r[(j+d)&15], r[(j+7)&15]);
    }
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; j++) s += r[j];
    out[i] = s;
}
"""
flops_reg = 16 * 64 * 2 * N
print("\n### Exhibit 4: hidden local memory (TRAP — only the compile stat reveals it)")
reg_free = cp.RawKernel(reg_src, "reg_heavy")
t = measure(reg_free, (GRID,), (BLOCK,), (a_in, a_out, N))
report("reg_heavy (static index)", flops_reg / (8 * N), 8 * N, flops_reg, t, reg_free,
       "compute-bound", note="array indexed by compile-time j → stays in registers")
reg_local = cp.RawKernel(reg_src_local, "reg_heavy_local")
t = measure(reg_local, (GRID,), (BLOCK,), (a_in, a_out, N))
report("reg_heavy (runtime index)", flops_reg / (8 * N), 8 * N, flops_reg, t, reg_local,
       "compute-bound", note="one-char change: r[(j+d)] with d runtime → forced to local mem")

# === Exhibit 5: uncoalesced TRAP — same source shape as SAXPY, strided index ==
stride_src = r"""
extern "C" __global__ void saxpy_strided(const float* a, const float* b, float* c, int n) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int i = (int)(((long)t * 131ul) % (unsigned long)n);   // scatter consecutive threads
    if (t < n) c[i] = 2.0f*a[i] + b[i];
}
"""
stride_kernel = cp.RawKernel(stride_src, "saxpy_strided")
print("\n### Exhibit 5: coalescing (TRAP — invisible in source; only BW reveals it)")
t = measure(stride_kernel, (GRID,), (BLOCK,), (a_in, b, c, N))
report("saxpy_strided", 2 / 12, 12 * N, 2 * N, t, stride_kernel, "HBM-bound",
       note="compare its GB/s to Exhibit 2's — same source, scattered addresses")

print("\n" + "=" * 78)
print("Read the TRAP lines: did L2-reuse exceed HBM peak? did the capped kernel")
print("spill (local>0) and slow down? did strided GB/s fall well below SAXPY's?")
