"""profile_decompress.py — characterize the decompression kernels into the
Phase 2 regimes (compute-bound / HBM-bound / latency-bound), two ways:

  1. cupy timing  -> *inferred* regime from apparent bandwidth + occupancy
                     (same classifier as gpu_kernel_regimes.py). Fast, no tools.
  2. Nsight Compute -> *confirmed* regime from real counters, and — crucially —
                     it resolves the single "latency-bound" label into its cause
                     (register spill vs uncoalesced access vs low ILP vs L2-served).

The kernels here are the 32-MiB decompression **HBM-roof floor** (memory traffic
of decompression, none of the Huffman/Lorenzo math). Paste your real decompress
kernel into KERNELS to profile it the same way.

Usage (on the A100 node):
    uv run python profile_decompress.py            # inferred regimes
    uv run python profile_decompress.py --ncu      # counter-confirmed regimes
    uv run python profile_decompress.py --list     # list registered kernels

Requires a CUDA GPU + cupy. `--ncu` additionally needs `ncu` on PATH and GPU
performance-counter permissions (on shared nodes you may need the admin to grant
them, or run under `sudo`; see NVIDIA ERR_NVGPUCTRPERM).
"""

import argparse
import csv
import io
import subprocess
import sys

import cupy as cp
from cupyx.profiler import benchmark

import regimes

# --- one event ---------------------------------------------------------------
N = 1 << 23        # 8,388,608 floats = exactly 32 MiB output (one event)
BLOCK = 256
EXPAND = 10        # compression ratio ~10x -> compressed input is 3.2 MiB
COMP_N = N // EXPAND
GRID = ((N + BLOCK - 1) // BLOCK,)


WRITE_SRC = r"""
extern "C" __global__ void hbm_floor_write(float* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f;                 // 32 MiB coalesced write, no math
}
"""

RW_SRC = r"""
extern "C" __global__ void hbm_floor_rw(const float* comp, int comp_n,
                                        float* out, int n, int expand) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float code = comp[i / expand];            // 3.2 MiB read once from DRAM (L2 after)
    out[i] = code;                            // 32 MiB coalesced write
}
"""


def _buffers():
    return {
        "out": cp.empty(N, dtype=cp.float32),
        "comp": cp.ones(COMP_N, dtype=cp.float32),
    }


# Each kernel carries how to build its args, its *source-counted* bytes (what the
# arithmetic-intensity story counts), and its expected *DRAM* bytes (what actually
# crosses the bus — differs when reads are L2-served, the Exhibit-3 effect).
KERNELS = [
    {
        "name": "hbm_floor_write", "entry": "hbm_floor_write", "src": WRITE_SRC,
        "args": lambda b: (b["out"], N),
        "src_bytes": 4 * N, "dram_bytes": 4 * N, "flops": 0,
        "note": "output-write floor",
    },
    {
        "name": "hbm_floor_rw", "entry": "hbm_floor_rw", "src": RW_SRC,
        "args": lambda b: (b["comp"], COMP_N, b["out"], N, EXPAND),
        "src_bytes": 8 * N, "dram_bytes": 4 * N + 4 * COMP_N, "flops": 0,
        "note": "read-compressed + write-output floor",
    },
    # --- paste your real decompressor here -----------------------------------
    # {
    #     "name": "decompress_real", "entry": "decompress_real", "src": REAL_SRC,
    #     "args": lambda b: (b["comp"], COMP_N, b["out"], N),
    #     "src_bytes": 4 * COMP_N + 4 * N,   # bytes your source *thinks* it moves
    #     "dram_bytes": 4 * COMP_N + 4 * N,  # best-guess actual DRAM traffic
    #     "flops": 8 * N,                    # ~FLOPs of dequant + Lorenzo per element
    #     "note": "real Huffman + Lorenzo",
    # },
]
KERNEL_BY_NAME = {k["name"]: k for k in KERNELS}


# ==============================================================================
# Mode 1 — cupy timing -> inferred regime
# ==============================================================================

def profile_timing(peaks):
    bufs = _buffers()
    print(f"GPU: {peaks['name']}  peak FP32 {peaks['peak_fp32_gflops']/1e3:.1f} TFLOP/s  "
          f"HBM {peaks['peak_hbm_gbps']:.0f} GB/s  ridge {regimes.ridge(peaks):.1f}\n")
    print(f"event: 32 MiB out ({N/1e6:.1f}M floats), grid {GRID[0]} blocks x {BLOCK}\n")

    for k in KERNELS:
        kernel = cp.RawKernel(k["src"], k["entry"])
        _ = kernel.num_regs  # force NVRTC compile so the compile stats populate
        args = k["args"](bufs)
        t = benchmark(lambda: kernel(GRID, (BLOCK,), args),
                      n_repeat=50, n_warmup=10).gpu_times.mean()

        gflops = k["flops"] / t / 1e9
        src_gbps = k["src_bytes"] / t / 1e9      # apparent BW from source byte count
        dram_gbps = k["dram_bytes"] / t / 1e9    # BW from bytes that truly hit DRAM
        frac_fl = gflops / peaks["peak_fp32_gflops"]
        frac_bw = dram_gbps / peaks["peak_hbm_gbps"]
        # Classify on the DRAM traffic (the honest regime); the source-counted BW is
        # reported alongside so an L2-served read shows up as apparent > 100% HBM.
        verdict = regimes.classify(frac_bw, frac_fl, dram_gbps, peaks["peak_hbm_gbps"])
        occ = regimes.occupancy(kernel.num_regs, kernel.shared_size_bytes, peaks)

        print(f"[{k['name']}]  {k['note']}")
        print(f"  compile:  regs/thread={kernel.num_regs}  smem={kernel.shared_size_bytes}B  "
              f"spill(local)={kernel.local_size_bytes}B")
        print(f"  occupancy: {occ['blocks_per_sm']} blocks/SM = {occ['occ_pct']:.0f}%  "
              f"(binding cap: {occ['binding']})")
        print(f"  timing:   {t*1e6:6.1f} us/event  ->  {1/t:8.0f} events/s")
        print(f"  DRAM BW:  {dram_gbps:6.0f} GB/s ({frac_bw*100:5.1f}% HBM)   "
              f"source-counted BW: {src_gbps:.0f} GB/s "
              f"({src_gbps/peaks['peak_hbm_gbps']*100:.0f}% HBM)")
        print(f"  INFERRED REGIME: {verdict}\n")

    print("Note: cupy timing infers the regime from apparent bandwidth. It CANNOT\n"
          "tell apart the causes of 'latency-bound' (spill / uncoalesced / low ILP /\n"
          "L2-served). Re-run with --ncu for counter-backed characterization.")


# ==============================================================================
# Mode 2 — Nsight Compute -> confirmed regime + resolved latency cause
# ==============================================================================

# Metrics chosen to map onto the regime table. Roof utilizations decide the coarse
# regime; the rest resolve *why* a kernel is latency-bound.
NCU_METRICS = [
    "gpu__time_duration.sum",                                   # kernel duration
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",        # compute roof utilization
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",      # HBM roof utilization
    "sm__warps_active.avg.pct_of_peak_sustained_active",       # achieved occupancy
    "dram__bytes.sum",                                         # bytes that truly hit DRAM
    "lts__t_sector_hit_rate.pct",                             # L2 hit rate (cache-served tell)
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",          # local-mem reads  (spill tell)
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",          # local-mem writes (spill tell)
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",  # load coalescing %
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",  # store coalescing %
]


def _num(s):
    """ncu --csv wraps numbers in quotes and uses thousands separators."""
    s = s.strip().replace(",", "")
    if s in ("", "N/A", "n/a"):
        return 0.0
    return float(s)


def parse_ncu_csv(text):
    """Return {kernel_name: {metric_name: value}} from `ncu --csv` stdout."""
    out = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        kname = row.get("Kernel Name")
        mname = row.get("Metric Name")
        if not kname or not mname:
            continue
        out.setdefault(kname, {})[mname] = _num(row.get("Metric Value", "0"))
    return out


def characterize(m):
    """Map a metric dict to a regime + (for latency-bound) its resolved cause."""
    sm = m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0.0)
    dram = m.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0.0)
    occ = m.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0)
    l2_hit = m.get("lts__t_sector_hit_rate.pct", 0.0)
    local = (m.get("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", 0.0)
             + m.get("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum", 0.0))
    ld_eff = m.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", 100.0)
    st_eff = m.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct", 100.0)

    # Coarse regime from which roof is actually saturated (thresholds are heuristic).
    if dram >= 60 and dram >= sm:
        return "HBM-bound", []
    if sm >= 60 and sm > dram:
        return "compute-bound", []

    # Neither roof saturated -> latency-bound. Resolve the cause(s) from counters.
    causes = []
    if local > 0:
        causes.append(f"register spill — {local:.0f} local-memory sectors (should be 0)")
    if min(ld_eff, st_eff) < 50:
        causes.append(f"uncoalesced global access — ld {ld_eff:.0f}% / st {st_eff:.0f}% "
                      "bytes-per-sector efficiency")
    if occ < 30:
        causes.append(f"low occupancy — {occ:.0f}% achieved warps active")
    if l2_hit > 60:
        causes.append(f"cache-served — {l2_hit:.0f}% L2 hit rate (traffic not really DRAM)")
    if not causes:
        causes.append("instruction/dependency latency — low ILP, both roofs idle "
                      "(e.g. a dependent predictor/Huffman chain)")
    return "latency-bound", causes


def profile_ncu():
    for k in KERNELS:
        cmd = [
            "ncu", "--csv", "--metrics", ",".join(NCU_METRICS),
            "--kernel-name", k["entry"], "--launch-skip", "3", "--launch-count", "1",
            "--target-processes", "all",
            sys.executable, __file__, "--run-one", k["name"],
        ]
        # ncu absent / no permission is the expected failure mode on a fresh node.
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            print("ncu not found on PATH. Load the CUDA toolkit / Nsight Compute module "
                  "and retry, or drop --ncu for the timing-only inference.")
            return

        if proc.returncode != 0:
            print(f"[{k['name']}] ncu failed (rc={proc.returncode}):\n{proc.stderr.strip()}\n")
            continue

        parsed = parse_ncu_csv(proc.stdout)
        m = parsed.get(k["entry"]) or (next(iter(parsed.values())) if parsed else {})
        if not m:
            print(f"[{k['name']}] no metrics parsed — raw ncu output:\n{proc.stdout}\n")
            continue

        regime, causes = characterize(m)
        dur_us = m.get("gpu__time_duration.sum", 0.0) / 1e3  # ns -> us
        print(f"[{k['name']}]  {k['note']}")
        print(f"  duration: {dur_us:.1f} us   "
              f"DRAM {m.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 0):.0f}% roof   "
              f"SM {m.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 0):.0f}% roof   "
              f"occ {m.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0):.0f}%")
        print(f"  DRAM bytes (measured): {m.get('dram__bytes.sum', 0)/1e6:.1f} MB  "
              f"vs source-counted {k['src_bytes']/1e6:.1f} MB")
        print(f"  CONFIRMED REGIME: {regime}")
        for c in causes:
            print(f"    - cause: {c}")
        print()


def run_one(name):
    """Launch a single kernel a few times so `ncu --launch-skip 3 -c 1` can attach
    to a warmed-up steady-state launch. Only reached under ncu, so no GPU guard."""
    k = KERNEL_BY_NAME[name]
    bufs = _buffers()
    kernel = cp.RawKernel(k["src"], k["entry"])
    args = k["args"](bufs)
    for _ in range(5):
        kernel(GRID, (BLOCK,), args)
    cp.cuda.runtime.deviceSynchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ncu", action="store_true", help="characterize via Nsight Compute counters")
    ap.add_argument("--list", action="store_true", help="list registered kernels and exit")
    ap.add_argument("--run-one", metavar="NAME", help="(internal) one warmed launch for ncu")
    args = ap.parse_args()

    if args.list:
        for k in KERNELS:
            print(f"{k['name']:20} {k['note']}")
        return
    if args.run_one:
        run_one(args.run_one)
        return
    if args.ncu:
        profile_ncu()
        return
    profile_timing(regimes.detect_peaks())


if __name__ == "__main__":
    main()
