# Prototype verdict — "tells vs traps" kernel gallery

**Question:** For Phase 2 (`gpu_kernel_regimes.py`), which source-level regime
tells are reliable, and which are *traps* that only measurement reveals? Each
trap is only worth a notebook cell if it actually diverges from its
source-predicted regime on the A100.

Run: `uv run python prototype_regime_tells.py` (SLAC Ampere node).

## Expected, per exhibit

| Exhibit | Source prediction | What confirms it reproduced |
|---|---|---|
| 1 · AI-dial | HBM at low iters → compute at high | GFLOP/s plateaus; verdict flips across `iters` near the ridge |
| 2 · SAXPY | HBM-bound | reads HBM-bound, near peak GB/s |
| 3 · L2-reuse | (source) HBM-bound | **TRAP** if apparent GB/s > HBM peak ⇒ "cache-served" |
| 4 · register spill | compute-bound | **TRAP** if capped build shows `local>0` and slows down |
| 5 · uncoalesced | HBM-bound | **TRAP** if strided GB/s ≪ Exhibit 2's GB/s |

## VERDICT (run on NVIDIA A100-SXM4-40GB, CC 8.0, 108 SMs)

Detected peaks: FP32 ~19.5 TFLOP/s, HBM ~1555 GB/s, ridge ~12.5 FLOPs/byte.

- **AI-dial control — WORKS.** Verdict flips HBM→compute across the ridge: AI=2
  reads HBM-bound (72% HBM), AI=16 reads compute-bound. *Nuance:* compute end
  tops out at ~39% of peak FP32 because the single dependent FMA chain is
  **latency-bound** (low ILP), not throughput-bound. Worth its own callout.
- **SAXPY control — WORKS.** Clean HBM-bound at 83% of peak bandwidth.
- **L2-reuse trap — REPRODUCED (star exhibit).** Apparent 2726 GB/s = **175% of
  HBM peak**, physically impossible → traffic served from the 40 MB L2, not DRAM.
  Source AI says HBM-bound; reality is cache-served. Clean, dramatic.
- **Register-spill trap — PIVOTED.** `--maxrregcount` (NVRTC path) and
  `__launch_bounds__` both **failed to bind** through CuPy's JIT (SASS stayed at
  23 regs, no spill). Replaced with a **dynamically-indexed private array**: a
  runtime index into `r[]` forces it into local memory (registers aren't
  addressable — a hard rule, not a heuristic). Result: `local=64B` and a **13×
  slowdown** (18211 → 1362 GFLOP/s) from a one-character source change. The only
  tell is the `local_size_bytes` compile stat. Most dramatic exhibit.
- **Uncoalesced trap — REPRODUCED, reframe.** Strided SAXPY drops 1291 → 65 GB/s
  (~20×) vs the coalesced version. Both stay "HBM-bound" — the regime label does
  NOT flip — so frame it as *"coalescing decides whether you reach 83% or 4% of
  the roof,"* not as a misclassification.

- **Shared-memory exhibit (Ex6, added after Q) — WORKS.** `__shared__ float
  tile[8192]` (32 KB/block) reports `smem=32768B`, and the occupancy ceiling's
  **binding cap flips from `thread` to `smem`: 5 blocks/SM = 62%** (164 KB/SM ÷
  32 KB). Demonstrates both faces of shared memory — deliberate on-chip reuse
  (the chosen cousin of the L2 trap) *and* its occupancy cost — and ties directly
  to the Phase 1 `min(thread, block, reg, smem)` ceiling. Verdict reads
  latency-bound (smem-access latency at reduced occupancy); the headline is the
  cap flip, not the regime label.

**Decision for the demo:** keep **all six**. Two structural findings:

1. **Three regimes, not two.** Compute-bound, bandwidth(HBM)-bound, and
   **latency-bound** — the last is "under neither roof" (low ILP / dependent
   chains, uncoalesced access, register spills to local). The 2-axis roofline
   can't express it; occupancy + compile stats can. The classifier flags it when
   `max(frac_bw, frac_fl) < 0.25`.
2. **Spine is broader than "predict the regime."** Ex4/Ex5/Ex6 don't flip a clean
   label — they collapse performance or occupancy. Thesis: **"read the source to
   form a hypothesis; the compile stats (regs/smem/local) + occupancy + a
   measurement are what confirm or destroy it."** Every exhibit now prints an
   occupancy line (binding cap), reusing the Phase 1 ceiling math.

Sub-lesson from Ex1: compute-bound ≠ peak — ILP/latency matters (40%, not 100%).

Toolchain finding to carry into the real notebook: register caps via
`--maxrregcount`/`__launch_bounds__` are unreliable through CuPy RawKernel JIT
(SASS stayed at 23 regs); use dynamic indexing (forces local) or
`backend='nvcc'` if a hard reg cap is ever needed.

_Next: fold these five exhibits into `gpu_kernel_regimes.py`, then delete
`prototype_regime_tells.py`._
