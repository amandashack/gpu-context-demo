# Handoff — Phase 2 demo: kernel regimes and tuning tradeoffs

**For:** the agent starting the *next* demo, which shifts focus from "how do
N workers share one GPU?" to **"how does the kernel itself determine which
regime you're in, and which tuning knob actually matters in each regime?"**

This is the demo that answers the question Phase 1 deliberately doesn't:
*which bound condition does my kernel hit, and what do I tune to move it?*

## Read these first

- `docs/handoff-phase1-cleanup.md` (the companion handoff). That handoff has
  open items that should land before this demo starts — especially the honest
  occupancy-ceiling calculation and ADR-0002 (the Phase 2 scoping ADR).
- `CONTEXT.md` — canonical vocabulary. Reuse Worker / Run / Trial / Sweep /
  Occupancy / Launch-bound-vs-compute-bound. Do not invent parallel terms.
- `docs/adr/0001-nvshmem-out-of-scope.md` — boundary on multi-GPU scope.
- `/home/ajshack/.claude/projects/-home-ajshack-personal/memory/MEMORY.md`
  — auto-memory index. The user is comfortable with MPI/distributed but newer
  to CUDA jargon, so define on first use. Use `uv` for env management; never
  pip/venv/poetry/conda. Per `~/.claude/CLAUDE.md`, **no `try` statements
  unless a failure is genuinely expected** (and then comment why).
- Coworker's overlap notes at
  `psana/psana/gpu/multiowner/cuda_overlap_model_notes.md` on the
  `features/psana2-gpu-multiowner-baseline` branch of `slac-lcls/lcls2`.
  This demo will reach into the same regime the overlap notes already cover
  (real H2D/D2H), so coordinate naming and avoid duplicating their work —
  build on it, don't reinvent it.

## The motivating gap

Phase 1's `busy_kernel` is pure FMA: no global-memory traffic, no copies.
It is *intentionally* compute-bound, and the demo's two axes (occupancy and
`work_per_thread`) both stay inside the compute-bound regime. As a result the
Phase 1 demo cannot show:

- The HBM-bound regime (kernel stalls on device DRAM)
- The transfer-bound regime (PCIe H2D saturated)
- The tuning levers that change which regime you sit in (registers/thread,
  shared memory/block, arithmetic intensity)

The user has internalised this and explicitly asked the next demo to expose
**different regimes and the tradeoffs that move you between them**.

## The three bound conditions to demonstrate

| condition | what's saturated | typical fix |
|---|---|---|
| **Compute-bound** | SM functional units | already the headline regime in Phase 1; carry it forward as the reference point |
| **HBM-bound** | device DRAM bandwidth (~1.5 TB/s A100) | raise arithmetic intensity, tile in shared memory, fuse kernels |
| **Transfer-bound (H2D-bandwidth-bound)** | PCIe link (Gen4 ×16 ≈ 25 GB/s) | overlap H2D with compute (see coworker's notes); cannot be parallelized away — copy engines are a hard ceiling |

These are conceptually distinct — *transfer-bound* is **not** the same as
"memory-bound" in the standard CUDA sense, and the doc must keep this
terminology straight or the reader will conflate HBM and PCIe.

## The tuning knobs to expose

Each knob should be a control in the notebook with a visible consequence.
Conversation already covered the underlying math; treat that as the lecture
the demo replaces with an interactive view.

- **Registers per thread.** Three ways to influence it:
  1. Kernel code (more live variables → more regs)
  2. `nvcc -maxrregcount=N` hard cap
  3. `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`
     hint

  The tradeoff: lower regs/thread raises the *occupancy ceiling* but can
  cause **spills to local memory** (per-thread private storage backed by
  L1/L2/HBM, ~30+ cycle access vs ~1 cycle for registers). Demoing a kernel
  where dropping from e.g. 64 → 32 regs/thread doubles occupancy but halves
  throughput is the right shape — that proves "higher occupancy is not
  always faster."

- **Shared memory per block.** Tighter the on-chip cooperative storage, the
  more blocks fit per SM. But too little kills tiling efficiency. Same
  curve shape as regs/thread.

- **threads_per_block.** Held fixed at 256 in Phase 1 (CONTEXT.md:75).
  Phase 2 should let it vary — affects warp count, occupancy granularity,
  and the register-pressure × block-count interaction.

- **Arithmetic intensity (FLOPs/byte).** The kernel's position on the
  roofline. A100 FP32 ridge point ≈ 13 FLOPs/byte (≈ 78 for FP16). Below
  the ridge → HBM-bound; above → compute-bound. A demo kernel with a
  tunable AI dial (e.g. an FMA loop inside a memory-read loop with a
  configurable inner trip count) is the cleanest way to show the roofline
  knee.

## The occupancy ceiling — get it right this time

Phase 1's ceiling is thread-count-only. Phase 2 must compute the real one:

```text
blocks_per_SM = min(
    threads_per_SM // threads_per_block,                  # thread cap
    max_blocks_per_SM,                                    # block cap (32 on A100)
    regs_per_SM // (threads_per_block * regs_per_thread), # register cap
    smem_per_SM // smem_per_block,                        # shared-mem cap
)
occupancy_pct = blocks_per_SM * threads_per_block / threads_per_SM
```

Inputs:

- `kernel.num_regs` and `kernel.shared_size_bytes` from the compiled CuPy
  `RawKernel`.
- `cp.cuda.runtime.getDeviceProperties(0)` for SM count, threads per SM,
  registers per SM, shared memory per SM, max blocks per SM.

A100 reference numbers (also useful for the ADR):

- SMs: 108
- Threads/SM: 2048
- **Registers/SM: 65,536 × 32-bit (256 KB)** — *not* 2048, common confusion
- Shared mem/SM: ~164 KB
- Max blocks/SM: 32
- Compute capability: 8.0

## Naming / scope decisions to confirm with the user

These were left open in conversation. Surface them before writing code.

- **New notebook vs extending `gpu_demo.py`?** Conversation tilted toward
  a new notebook (e.g. `gpu_kernel_regimes.py`) so the Phase 1 demo stays
  the clean substrate-sharing artifact. Confirm.
- **Naming the "modes" in Phase 2.** The Mode A/B/C labels belong to the
  sharing-mechanism axis. Phase 2 needs *new* axis names — probably
  *regime* (compute/HBM/transfer) and *tune* (regs, smem, tpb, AI). Don't
  reuse "Mode" for both.
- **Will Phase 2 also re-vary mode A/B/C?** Probably yes (because real
  pipelines with H2D/D2H still pick a sharing mechanism), but the cube
  (mode × regime × tune × N) is large. Recommend: lock mode = C for the
  first pass and re-introduce A/B as a later overlay, matching the
  coworker's existing scope.
- **Real H2D/D2H means real input data.** Decide buffer sizes early — the
  psana2 target is 32 MB/event; Phase 2 should at least *include* that
  regime so the demo speaks to the production case.

## Connection back to the lcls2 pipeline

The driving production case: 32 MB events at high rate. The A100 H2D
ceiling on PCIe Gen4 ×16 is

```text
~25 GB/s / 32 MB  =  ~780 events/sec  (aggregate, all workers combined)
```

Past that the pipeline is **H2D-bandwidth-bound** regardless of how the
kernel itself is characterised. The coworker's overlap notes already show
the in-pipeline mechanics for hiding H2D behind compute (direct vs
row-based stream layout, simple vs advanced buffer reuse). Phase 2's value
is **connecting kernel-level regime to pipeline-level behaviour**: when the
kernel is compute-bound, overlap fully hides the copy and you're rate-limited
by SMs; when the kernel is HBM-bound at low AI, overlap helps less because
the GPU is already stalled on its own memory traffic.

A useful closing visualisation: a roofline plot with the kernel's current
operating point overlaid, plus a marker for "where the rate ceiling at
780 events/sec would put you on the roofline" — answers *do I have headroom
on the GPU, or am I PCIe-blocked already?*

## Suggested skills

- `prototype` — exactly the situation it's designed for. Build a throwaway
  prototype (a small Marimo notebook with one or two synthetic kernels) to
  flush out the design — knob layout, plot shape, how the
  regime-transitions read — before committing to the real demo.
- `grill-with-docs` — once a plan exists, stress-test it against the
  CONTEXT.md vocabulary and ADR-0001/0002 before any code. The user has
  used this pattern productively and likes to confirm naming and scope up
  front rather than discovering conflicts mid-implementation.
- `improve-codebase-architecture` — only after both demos exist and there's
  shared infrastructure (kernel compilation, device-property lookup,
  worker pool) worth pulling into common code.
