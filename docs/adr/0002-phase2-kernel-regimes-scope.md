# Phase 2 scope — kernel regimes and tuning tradeoffs

Phase 1 answered *"should N workers share one GPU, and when does sharing stop
helping?"* using a deliberately compute-bound kernel (`busy_kernel`, pure FMA).
Phase 2 answers the question Phase 1 cannot: **which bound condition does my
kernel actually hit, and which tuning knob moves it?** This ADR fixes the scope
of that second demo so the work doesn't sprawl across the full
mode × regime × tune × N cube.

## New notebook, not an extension

Phase 2 lives in a **new notebook, `gpu_kernel_regimes.py`**. `gpu_demo.py`
stays the clean Phase 1 context-sharing artifact; mixing regime/tuning views
into it would blur the one question each demo is meant to answer. Shared
infrastructure (device-property lookup, the honest occupancy-ceiling math,
kernel compilation) can be pulled into a common module *after* both demos exist
— not preemptively.

## The linchpin: a tunable memory-traffic kernel

Every Phase 2 thread depends on one new artifact: **a kernel with a tunable
arithmetic intensity (AI) dial** — an FMA loop wrapped around a configurable
amount of global-memory traffic (e.g. a memory-read loop with an adjustable
inner FMA trip count). This is the linchpin; build and validate it before
anything else. With it, sliding AI walks the kernel across the roofline knee;
without it there is no HBM-bound regime to show. `busy_kernel` carries forward
unchanged as the compute-bound reference point (AI → ∞).

## Axes and naming

The Phase 1 `Mode` A/B/C labels belong to the **sharing-mechanism** axis and are
not reused. Phase 2 introduces two new axes:

- **regime** — `compute` / `HBM` / `transfer`, the bound condition the kernel
  sits in.
- **tune** — the lever being varied: registers/thread, shared-memory/block,
  `threads_per_block`, arithmetic intensity.

Keeping "regime" and "tune" distinct from "mode" is a hard naming rule; the
reader must never conflate the PCIe-bound *transfer* regime with the standard
CUDA *memory-bound* (HBM) regime, so the doc spells those out separately.

## Sharing mode locked to B (MPS) for the first pass

The first pass **fixes the sharing mechanism at Mode B (MPS)** rather than
sweeping A/B/C. Rationale: the driving production case (lcls2 psana2) is N
worker *processes* sharing one GPU, which is exactly what MPS models — Mode C's
single-process streams is the less representative mechanism here, and the SLAC
Ampere nodes have the MPS daemon available (unlike the WSL2 dev box where Phase 1
defaulted B off). Re-introducing A/C as an overlay is a later step, matching the
coworker's existing overlap-notes scope.

## Regime rollout: compute + HBM first, transfer second

1. **Compute + HBM first.** Build the roofline view driven by the AI dial. On
   A100 the FP32 ridge point is ≈ 13 FLOPs/byte (≈ 78 for FP16): below the ridge
   the kernel is HBM-bound (stalled on ≈ 1.5–2 TB/s device DRAM), above it
   compute-bound. This is the cleanest first deliverable and needs no host↔device
   copies.
2. **Transfer-bound second.** Add real H2D/D2H copies and the PCIe regime. The
   production target is 32 MB/event; on PCIe Gen4 ×16 (≈ 25 GB/s) the aggregate
   H2D ceiling is ≈ 780 events/sec regardless of kernel characterisation. This
   regime cannot be parallelized away — copy engines are a hard ceiling — and is
   where the demo connects kernel-level regime to pipeline-level behaviour, and
   to the coworker's copy/compute-overlap notes. Build on those, don't duplicate.

## Occupancy ceiling already landed

The honest occupancy ceiling —
`blocks_per_SM = min(thread, block, register, shared-mem caps)` from the
compiled kernel's `num_regs` / `shared_size_bytes` and device limits — was added
to `gpu_demo.py` in commit `c7ba95e` and is the calculation Phase 2 reuses. The
register- and shared-memory-pressure tuning knobs depend on it: dropping
regs/thread to raise the occupancy ceiling while watching throughput fall (from
local-memory spills) is the headline "higher occupancy is not always faster"
demonstration.

## Deferred to a later pass (still in Phase 2, not first cut)

- The full **mode × regime × tune** cube (re-varying A/C alongside regime).
- Alternative aggregation `sum(kernels) ÷ sum(elapsed)` vs `÷ max(elapsed)`.
- A second heatmap baselined on **Mode A at N=1** (scaling-efficiency lens).
- **Mode D** (1 process, N threads, N contexts) — still a diagnostic-only corner.

## Consequence

The "Flagged for Phase 2" lines previously scattered in `CONTEXT.md` collapse
into a single pointer to this ADR, so scope lives in one place.

A100 reference numbers used above (the SLAC Ampere target, CC 8.0): 108 SMs,
2048 threads/SM, 65,536 × 32-bit registers/SM (256 KB — *not* 2048, a common
confusion), ≈ 164 KB shared memory/SM, 32 blocks/SM max.
