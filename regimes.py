"""Shared logic for the Phase 2 kernel-regimes demo (`gpu_kernel_regimes.py`).

The pure, GPU-independent pieces live here — exhibit specs (the six CUDA
sources plus their source-counted FLOP/byte arithmetic), device-peak detection
with a cached A100 fallback, the honest occupancy ceiling, the regime
classifier, and cache I/O. The marimo notebook is a thin reactive shell over
these functions, and everything except `run_measurements` can be exercised
without an A100 (the committed `regime_cache.json` supplies the timings).

Terminology is kept in lockstep with the prototype (`prototype_regime_tells.py`)
and ADR-0002: *regime* is the bound condition (compute / HBM / latency), never
reused for the Phase 1 sharing *mode*.
"""

import json
import os

# --- workload sizes — identical to the prototype so cached numbers line up ----
N = 1 << 24        # 16M floats = 64 MB; exceeds the 40 MB L2 so streaming is real DRAM
M = 1 << 20        # 4 MB — fits comfortably in the 40 MB L2 (the reuse trap)
BLOCK = 256        # threads per block, held fixed across every exhibit
REUSE = 64         # inner re-read count for the L2 exhibit
AI_DIAL_ITERS = [1, 8, 64, 512]   # the cached sweep points for the AI dial

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regime_cache.json")

# CC -> FP32 cores/SM. A100 (8.0) is 64; the 8.6/8.9/9.0 parts double it.
FP32_CORES_PER_SM = {(7, 0): 64, (7, 2): 64, (7, 5): 64,
                     (8, 0): 64, (8, 6): 128, (8, 9): 128, (9, 0): 128}

# Cached A100-SXM4-40GB (CC 8.0) device constants, used when no GPU is present.
# Peaks match the prototype header (19.5 TFLOP/s, 1555 GB/s, ridge 12.5).
A100_CACHED = {
    "name": "NVIDIA A100-SXM4-40GB",
    "cc": "8.0",
    "sm": 108,
    "threads_per_sm": 2048,
    "regs_per_sm": 65536,
    "smem_per_sm": 167936,        # 164 KiB; // 32768 = 5 blocks/SM for the smem exhibit
    "max_blocks_per_sm": 32,
    "peak_fp32_gflops": 108 * 64 * 2 * 1.410e9 / 1e9,   # 19491.8
    "peak_hbm_gbps": 2 * 1.215e9 * (5120 / 8) / 1e9,    # 1555.2
    "cached": True,
}


# ==============================================================================
# CUDA sources — copied verbatim from prototype_regime_tells.py
# ==============================================================================

AI_DIAL_SRC = r"""
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

SAXPY_SRC = r"""
extern "C" __global__ void saxpy(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) c[i] = 2.0f*a[i] + b[i];
}
"""

L2_SRC = r"""
extern "C" __global__ void l2_reuse(const float* small, float* out, int m, int reuse) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float acc = 0.0f;
    #pragma unroll 1
    for (int k = 0; k < reuse; k++) acc += small[(i + k) % m];   // re-reads a small buffer
    out[i % m] = acc;
}
"""

REG_STATIC_SRC = r"""
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

# One-character change from REG_STATIC_SRC: the inner index into r[] is now
# RUNTIME-dependent (d). A register file isn't addressable, so any array indexed
# by a runtime value is forced into local memory -> local_size_bytes > 0. This is
# the toolchain-proof way to show hidden memory traffic the source doesn't
# advertise (register caps via --maxrregcount / __launch_bounds__ did NOT bind
# through CuPy's NVRTC JIT path).
REG_RUNTIME_SRC = r"""
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

STRIDED_SRC = r"""
extern "C" __global__ void saxpy_strided(const float* a, const float* b, float* c, int n) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int i = (int)(((long)t * 131ul) % (unsigned long)n);   // scatter consecutive threads
    if (t < n) c[i] = 2.0f*a[i] + b[i];
}
"""

SMEM_SRC = r"""
extern "C" __global__ void smem_reuse(const float* in, float* out, int n) {
    __shared__ float tile[8192];                 // 32 KB/block
    int t = threadIdx.x;
    int i = blockIdx.x*blockDim.x + t;
    for (int k = t; k < 8192; k += blockDim.x)   // stage block's data into smem once
        tile[k] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    float s = 0.0f;
    #pragma unroll 1
    for (int k = 0; k < 256; k++) s += tile[(t * 7 + k) & 8191];  // reuse from smem
    if (i < n) out[i] = s;
}
"""


# ==============================================================================
# Exhibit specs. Each entry is pure metadata: how to count source bytes/FLOPs
# (so arithmetic intensity is known without a GPU), what regime the *source*
# predicts, and a one-line teaching hook. `iters` is only meaningful for the
# AI dial. `variant_of` groups the two register kernels into one exhibit card.
# ==============================================================================

def _ai_counts(iters):
    """AI-dial: read+write one float, `iters` FMAs (2 FLOPs each)."""
    src_bytes = 8 * N
    flops = 2 * N * iters
    return flops / src_bytes, src_bytes, flops


EXHIBITS = [
    {
        "key": "ai_dial", "num": 1, "title": "AI-dial", "category": "control",
        "entry": "ai_dial", "src": AI_DIAL_SRC, "swept": True,
        "hook": "Sweep `iters` and the operating point walks up the memory roof, "
                "rounds the ridge, and flattens under the compute roof. The one "
                "exhibit where the source-level roofline prediction actually holds.",
        "caveat": "The compute end tops out near ~35% of peak FP32, not 100% — the "
                  "single dependent FMA chain is FP-latency-bound (low ILP), so even "
                  "the 'clean' control predicts *which roof*, never *how close* you get.",
    },
    {
        "key": "saxpy", "num": 2, "title": "SAXPY", "category": "control",
        "entry": "saxpy", "src": SAXPY_SRC, "swept": False,
        "src_ai": 2 / 12, "src_bytes": 12 * N, "flops": 2 * N, "predicted": "HBM-bound",
        "hook": "Textbook HBM-bound (AI ~0.17). The coalesced reference point that "
                "gives Exhibit 5 its meaning — same source, opposite bandwidth.",
    },
    {
        "key": "l2_reuse", "num": 3, "title": "L2-reuse", "category": "trap",
        "entry": "l2_reuse", "src": L2_SRC, "swept": False,
        "src_ai": (N * REUSE) / (4 * N * REUSE), "src_bytes": 4 * N * REUSE,
        "flops": N * REUSE, "predicted": "HBM-bound",
        "hook": "The source counts every re-read as a DRAM byte, so it predicts "
                "HBM-bound. But the 4 MB buffer lives in the 40 MB L2 — apparent "
                "bandwidth exceeds the DRAM peak, which is physically impossible.",
    },
    {
        "key": "reg_static", "num": 4, "title": "Register array (static index)",
        "category": "control", "variant_of": "reg_spill",
        "entry": "reg_heavy", "src": REG_STATIC_SRC, "swept": False,
        "src_ai": (16 * 64 * 2 * N) / (8 * N), "src_bytes": 8 * N,
        "flops": 16 * 64 * 2 * N, "predicted": "compute-bound",
        "hook": "Array indexed by a compile-time constant → stays in registers → "
                "compute-bound at ~93% of peak, as the source predicts.",
    },
    {
        "key": "reg_runtime", "num": 4, "title": "Register array (runtime index)",
        "category": "trap", "variant_of": "reg_spill",
        "entry": "reg_heavy_local", "src": REG_RUNTIME_SRC, "swept": False,
        "src_ai": (16 * 64 * 2 * N) / (8 * N), "src_bytes": 8 * N,
        "flops": 16 * 64 * 2 * N, "predicted": "compute-bound",
        "hook": "One-character change — `r[(j+d)]` with `d` runtime-dependent — "
                "forces the array to local memory (`spill > 0`) and throughput "
                "craters. Same source AI, opposite outcome; only the compile stat tells.",
    },
    {
        "key": "saxpy_strided", "num": 5, "title": "Uncoalesced SAXPY",
        "category": "trap", "entry": "saxpy_strided", "src": STRIDED_SRC, "swept": False,
        "src_ai": 2 / 12, "src_bytes": 12 * N, "flops": 2 * N, "predicted": "HBM-bound",
        "hook": "Same source shape as Exhibit 2, but `i = (t*131) % n` scatters "
                "consecutive threads across memory. Coalescing is a runtime address "
                "property — invisible in the arithmetic — and it decides whether you "
                "reach 83% or 4% of the roof.",
    },
    {
        "key": "smem_reuse", "num": 6, "title": "Shared-memory reuse",
        "category": "tell", "entry": "smem_reuse", "src": SMEM_SRC, "swept": False,
        "src_ai": 256 / 8, "src_bytes": 8 * N, "flops": 256 * N, "predicted": "compute-bound",
        "hook": "32 KB/block of `__shared__` is deliberate on-chip reuse — the chosen "
                "cousin of the L2 trap. The tell is the occupancy ceiling: the binding "
                "cap flips from thread to smem (164 KB/SM ÷ 32 KB = 5 blocks/SM = 62%).",
        "caveat": "The printed source AI (32) mixes global-load bytes with smem "
                  "traffic that dominates the kernel, so it doesn't characterise what "
                  "the kernel actually does — read the occupancy cap, not the AI.",
    },
]

EXHIBIT_BY_KEY = {e["key"]: e for e in EXHIBITS}


def static_point(exhibit, iters=None):
    """Source-level (GPU-independent) numbers for an exhibit: (src_ai, bytes, flops, predicted)."""
    if exhibit.get("swept"):
        it = iters if iters is not None else AI_DIAL_ITERS[0]
        src_ai, src_bytes, flops = _ai_counts(it)
        predicted = "compute-bound" if src_ai > 12.5 else "HBM-bound"  # nominal ridge; peaks refine it
        return src_ai, src_bytes, flops, predicted
    return exhibit["src_ai"], exhibit["src_bytes"], exhibit["flops"], exhibit["predicted"]


# ==============================================================================
# Device peaks + occupancy
# ==============================================================================

def detect_peaks():
    """Return a peaks dict from the live GPU, or the cached A100 constants.

    cupy import / device query genuinely fails on a machine without a CUDA GPU
    (e.g. a laptop opening the notebook), which is the expected fallback path —
    hence the guarded import.
    """
    try:
        import cupy as cp
    except Exception:  # no cupy / no CUDA runtime -> use the captured A100 numbers
        return dict(A100_CACHED)

    if cp.cuda.runtime.getDeviceCount() == 0:
        return dict(A100_CACHED)

    props = cp.cuda.runtime.getDeviceProperties(0)
    cc = (props["major"], props["minor"])
    cores = FP32_CORES_PER_SM.get(cc, 64)
    sm = props["multiProcessorCount"]
    clock_hz = props["clockRate"] * 1e3
    mem_hz = props["memoryClockRate"] * 1e3
    bus_bytes = props["memoryBusWidth"] / 8
    return {
        "name": props["name"].decode(),
        "cc": f"{cc[0]}.{cc[1]}",
        "sm": sm,
        "threads_per_sm": props.get("maxThreadsPerMultiProcessor", 2048),
        "regs_per_sm": props.get("regsPerMultiprocessor", 65536),
        "smem_per_sm": props.get("sharedMemPerMultiprocessor", 0),
        "max_blocks_per_sm": props.get("maxBlocksPerMultiProcessor", 32),
        "peak_fp32_gflops": sm * cores * 2 * clock_hz / 1e9,
        "peak_hbm_gbps": 2 * mem_hz * bus_bytes / 1e9,
        "cached": False,
    }


def ridge(peaks):
    """FLOPs/byte at the roofline knee."""
    return peaks["peak_fp32_gflops"] / peaks["peak_hbm_gbps"]


def occupancy(num_regs, smem_bytes, peaks, block=BLOCK):
    """Honest occupancy ceiling: blocks/SM is the smallest of four hardware caps.

    Mirrors the Phase 1 `min(thread, block, register, shared-mem)` calculation
    (gpu_demo.py) — a resource cap only binds when its per-kernel input is
    positive and the device exposes the limit.
    """
    caps = {
        "thread": peaks["threads_per_sm"] // block,
        "block": peaks["max_blocks_per_sm"],
    }
    if num_regs > 0:
        caps["reg"] = peaks["regs_per_sm"] // (block * num_regs)
    if smem_bytes > 0 and peaks["smem_per_sm"] > 0:
        caps["smem"] = peaks["smem_per_sm"] // smem_bytes
    blocks_per_sm = max(1, min(caps.values()))
    binding = min(caps, key=caps.get)
    occ_pct = blocks_per_sm * block / peaks["threads_per_sm"] * 100
    return {"caps": caps, "binding": binding, "blocks_per_sm": blocks_per_sm, "occ_pct": occ_pct}


# ==============================================================================
# The regime classifier — kept identical to prototype_regime_tells.py.
# (Sharpening the single "latency-bound" verdict for Ex4/5/6 is deferred; see
# the notebook's limitations section.)
# ==============================================================================

def classify(frac_bw, frac_fl, apparent_gbps, peak_hbm):
    if apparent_gbps > peak_hbm * 1.1:
        return "cache-served (NOT DRAM-bound)"   # >100% of HBM is physically impossible
    if max(frac_bw, frac_fl) < 0.25:
        return "latency-bound (saturates neither roof)"
    if frac_bw >= frac_fl:
        return "HBM-bound"
    return "compute-bound"


def build_record(exhibit, measured, peaks, iters=None):
    """Assemble a full display record from an exhibit spec + its raw measurement.

    `measured` is {gflops, apparent_gbps, num_regs, smem, spill} — from either a
    live benchmark or the cache. Everything else (fractions, verdict, occupancy,
    trap flag) is derived here so the live and cached paths render identically.
    """
    src_ai, src_bytes, flops, predicted = static_point(exhibit, iters)
    frac_bw = measured["apparent_gbps"] / peaks["peak_hbm_gbps"]
    frac_fl = measured["gflops"] / peaks["peak_fp32_gflops"]
    verdict = classify(frac_bw, frac_fl, measured["apparent_gbps"], peaks["peak_hbm_gbps"])
    occ = occupancy(measured["num_regs"], measured["smem"], peaks)
    return {
        "key": exhibit["key"],
        "num": exhibit["num"],
        "title": exhibit["title"],
        "category": exhibit["category"],
        "iters": iters,
        "src_ai": src_ai,
        "predicted": predicted,
        "gflops": measured["gflops"],
        "apparent_gbps": measured["apparent_gbps"],
        "frac_bw": frac_bw,
        "frac_fl": frac_fl,
        "num_regs": measured["num_regs"],
        "smem": measured["smem"],
        "spill": measured["spill"],
        "occ": occ,
        "verdict": verdict,
        # A trap = the source regime word doesn't appear in the measured verdict.
        "trap": predicted.split("-")[0] not in verdict,
    }


# ==============================================================================
# Cache I/O + live measurement
# ==============================================================================

def cache_key(exhibit_key, iters=None):
    return f"{exhibit_key}@{iters}" if iters is not None else exhibit_key


def load_cache(path=CACHE_PATH):
    """Return {cache_key: measured-dict} from the committed capture, or {} if absent."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)["measured"]


def all_records(peaks, measured_by_key):
    """Build the full record list the notebook plots, one per exhibit (AI-dial
    expands to one record per cached iters value)."""
    records = []
    for ex in EXHIBITS:
        if ex.get("swept"):
            for it in AI_DIAL_ITERS:
                m = measured_by_key.get(cache_key(ex["key"], it))
                if m is not None:
                    records.append(build_record(ex, m, peaks, iters=it))
        else:
            m = measured_by_key.get(cache_key(ex["key"]))
            if m is not None:
                records.append(build_record(ex, m, peaks))
    return records


def run_measurements(peaks):
    """Live A100 path: compile + benchmark every exhibit, returning a
    {cache_key: measured-dict} map ready for `all_records` / caching.

    Only reachable when a CUDA GPU is present (the notebook guards the call), so
    the cupy import here is not expected to fail.
    """
    import cupy as cp
    from cupyx.profiler import benchmark

    grid = ((N + BLOCK - 1) // BLOCK,)
    block = (BLOCK,)
    a_in = cp.ones(N, dtype=cp.float32)
    a_out = cp.empty(N, dtype=cp.float32)
    b = cp.ones(N, dtype=cp.float32)
    c = cp.empty(N, dtype=cp.float32)
    small = cp.ones(M, dtype=cp.float32)
    l2_out = cp.empty(M, dtype=cp.float32)

    def bench(kernel, args):
        _ = kernel.num_regs  # force the NVRTC compile so the stats below populate
        r = benchmark(lambda: kernel(grid, block, args), n_repeat=50, n_warmup=10)
        return r.gpu_times.mean(), kernel

    def measured(kernel, t_s, src_bytes, flops):
        return {
            "gflops": flops / t_s / 1e9,
            "apparent_gbps": src_bytes / t_s / 1e9,
            "num_regs": kernel.num_regs,
            "smem": kernel.shared_size_bytes,
            "spill": kernel.local_size_bytes,
        }

    out = {}

    ai_k = cp.RawKernel(AI_DIAL_SRC, "ai_dial")
    for it in AI_DIAL_ITERS:
        t, k = bench(ai_k, (a_in, a_out, N, it))
        _, sb, fl, _ = static_point(EXHIBIT_BY_KEY["ai_dial"], iters=it)
        out[cache_key("ai_dial", it)] = measured(k, t, sb, fl)

    simple = [
        ("saxpy", SAXPY_SRC, "saxpy", (a_in, b, c, N)),
        ("reg_static", REG_STATIC_SRC, "reg_heavy", (a_in, a_out, N)),
        ("reg_runtime", REG_RUNTIME_SRC, "reg_heavy_local", (a_in, a_out, N)),
        ("saxpy_strided", STRIDED_SRC, "saxpy_strided", (a_in, b, c, N)),
        ("smem_reuse", SMEM_SRC, "smem_reuse", (a_in, a_out, N)),
    ]
    for key, src, entry, args in simple:
        t, k = bench(cp.RawKernel(src, entry), args)
        _, sb, fl, _ = static_point(EXHIBIT_BY_KEY[key])
        out[cache_key(key)] = measured(k, t, sb, fl)

    t, k = bench(cp.RawKernel(L2_SRC, "l2_reuse"), (small, l2_out, M, REUSE))
    _, sb, fl, _ = static_point(EXHIBIT_BY_KEY["l2_reuse"])
    out[cache_key("l2_reuse")] = measured(k, t, sb, fl)

    return out


def save_cache(peaks, measured_by_key, path=CACHE_PATH):
    with open(path, "w") as f:
        json.dump({"peaks": peaks, "measured": measured_by_key}, f, indent=2)
