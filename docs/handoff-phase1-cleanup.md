# Handoff — Phase 1 cleanup & summary doc

**For:** the agent finishing the current Phase 1 demo polish, applying drafted
edits to the user's HackMD summary doc, and capturing the Phase 2 plan as an
ADR before the next demo starts.

## Where things live

- **Demo notebook:** `/home/ajshack/personal/gpu-context-demo/gpu_demo.py` —
  Marimo notebook, the canonical Phase 1 artifact. Run with
  `uv run marimo edit gpu_demo.py`.
- **Worker code:** `worker.py` — defines `KERNEL_SRC` (`busy_kernel`, a pure
  FMA loop with `#pragma unroll 1` — *deliberately compute-bound*) plus
  `run_worker` (process modes) and `run_streams` (Mode C).
- **Glossary / methodology:** `CONTEXT.md` — canonical definitions for Worker,
  Mode, Run, Sweep, Warmup, Occupancy/saturation, Launch-bound/compute-bound,
  and the "what's out of scope for Phase 1" list. Treat as the source of truth
  for vocabulary.
- **User-facing intro:** `README.md` — describes the three modes and how to
  read the heatmap.
- **ADRs:** `docs/adr/0001-nvshmem-out-of-scope.md`. Phase 2 plan should land
  as `0002-*`.
- **Dependencies:** `pyproject.toml`. uv-only; do not introduce
  pip/venv/poetry/conda. No `pyarrow`/`fastparquet` — sweep cache uses stdlib
  `json` (rows are plain dict-of-scalar). cupy-cuda12x is the GPU dep.

## Memory and user preferences to read first

- `/home/ajshack/.claude/projects/-home-ajshack-personal/memory/MEMORY.md` —
  index of user auto-memory. Especially:
  - `user_gpu_knowledge.md` — user is comfortable with MPI/distributed
    concepts, newer to CUDA terminology; define GPU jargon on first use.
  - `feedback_python_env.md` — always `uv`, never pip/venv/poetry/conda.
- `~/.claude/CLAUDE.md` — global rule: **no `try` statements except when a
  failure is genuinely expected, in which case add a comment explaining why.**

## Status of the demo

Phase 1 polish pass is largely done. Recent changes (verified at `ast.parse`
level only — visuals not yet re-run on a GPU node):

- Top intro rewritten to a "the question this answers" framing.
- "Key terms" accordion cell added (Worker / Occupancy / Saturation /
  Launch-bound / Compute-bound / Throughput / Speedup).
- Heatmap x-axis switched from raw `blocks` to **occupancy as % of
  saturation**, with raw block count carried in plotly `customdata` for
  hover.
- Color auto-fit now spans all N (not just the selected N) — fixes the
  cross-N rebasing trap where each N's heatmap was independently auto-scaled.
- Y-axis annotated as the launch/compute-bound dial.
- Sweep results persisted to `__marimo__/sweep_cache.json` so a kernel
  restart on the same node reloads instantly. **This does not survive an SSH
  drop on its own** — use tmux for that (runbook below).
- CONTEXT.md gained Occupancy/saturation and Launch-bound/compute-bound
  glossary entries.
- README.md "interesting regimes" line corrected (the win is at **low
  occupancy with longer kernels**, not "short kernels").

## What's missing from Phase 1

Three threads were started in conversation but **not yet applied**. The user
should be told what's pending and asked to confirm before each.

### 1. Two factual touch-ups for the user's HackMD summary doc

The HackMD doc lives outside this repo. Edits to it must be handed back to the
user as text to paste. The two corrections (drafted in conversation):

- **Register count:** "Each SM has a finite number of 32-bit registers
  (typically 2048)" is wrong. The 2048 is the max *threads* per SM. On A100 the
  register file is **65,536 × 32-bit registers per SM** (256 KB), with up to
  255 registers per thread.
- **Branching:** "branching... slows down the warp scheduler and forces it to
  serialize" should be *warp divergence* — when threads in the *same warp*
  take different paths, the warp executes both sides serially with inactive
  threads masked. The scheduler isn't slowed; the warp's effective throughput
  drops because it does both paths. Divergence across *different* warps is
  free.

### 2. Three drafted sections for the HackMD summary doc

Three sections to insert after "GPU Context Sharing Demo" and before
"definitions". Drafted in conversation (last revision removed the word
"substrate" in favor of "mode" because the user found "substrate" confusing):

- **Copy/Compute Overlap** — summarizes the coworker's experiment at
  `psana/psana/gpu/multiowner/cuda_overlap_model_notes.md` on the
  `features/psana2-gpu-multiowner-baseline` branch of slac-lcls/lcls2.
  Covers direct vs row-based stream layouts and simple vs advanced
  buffer-reuse.
- **Two axes of the same problem** — table showing the context demo varies
  *mode* at fixed work, the overlap notes vary *stream layout + reuse* at
  fixed mode (C).
- **Where this is heading** — calls out the H2D ceiling at 32 MB/event
  (~780 events/sec on PCIe Gen4 ×16), distinguishes
  *H2D-bandwidth-bound* (interconnect) from standard CUDA *memory-bound*
  (HBM), and frames the open next step.

The exact drafted text is in the conversation transcript — pull from there
and offer to revise rather than re-deriving from scratch.

### 3. Phase 1.x doc additions still open

- **"How to locate your kernel on the heatmap" section** in the notebook
  itself. Concrete recipe: compute per-kernel occupancy, measure duration
  with `cudaEventElapsedTime` / `cupyx.profiler.benchmark`, check arithmetic
  intensity for compute-vs-memory bound, then read the matching heatmap cell.
  This addresses the user's foundational concern: "I don't know how to answer
  *should I share the GPU at all?* without understanding my own kernel."
- **Honest occupancy ceiling.** The demo currently computes blocks/SM from
  threads only. Replace with
  `min(thread-cap, block-cap, register-ceiling, smem-ceiling)` using
  `kernel.num_regs` and `kernel.shared_size_bytes` plus device props
  (`cp.cuda.runtime.getDeviceProperties`). Worth doing before the next demo
  inherits the same calculation.

### 4. ADR-0002 — Phase 2 plan

Not yet written. Should capture the Phase 2 scope and identify the
**memory-traffic kernel variant as the shared linchpin** — every other Phase
2 thread depends on it. The threads to enumerate (currently scattered as
"Flagged for Phase 2" lines in `CONTEXT.md`):

- Memory-bound kernel variant with H2D/D2H copies (CONTEXT.md:69).
- Tier 2–3 occupancy view — register-pressure and shared-memory knobs,
  throughput-vs-occupancy plots.
- Scaling-efficiency heatmap baselined on Mode A at N=1 (CONTEXT.md:74–75).
- Alternative aggregation: `sum(kernels) ÷ sum(elapsed)` vs `÷ max(elapsed)`
  (CONTEXT.md:72–74).
- Mode D — 1 process, N threads, N contexts diagnostic (CONTEXT.md:24).

After ADR-0002 lands, **collapse the "Flagged for Phase 2" lines in
CONTEXT.md into a single pointer to the ADR** rather than leaving the
information in two places.

## Practical run / SSH-drop survival

Sweep cache only survives kernel restart on the same node. To survive a full
SSH disconnect, the user runs in tmux:

```bash
# On GPU host
tmux new -s demo
uv run marimo edit --headless --port 8080 --no-token gpu_demo.py
# Ctrl-b d to detach

# From laptop
ssh -L 8080:localhost:8080 <gpu-host>
# Open http://localhost:8080
```

If the user has not tested the polished demo on a GPU node yet, that's a
genuine open item — visuals are verified only at parse + dataflow level.

## Hand-off boundary with the *other* agent

The companion handoff
(`docs/handoff-phase2-kernel-regimes.md`) covers the *next* demo —
kernel-level regime exploration. Phase 1.x work above should land *before* it
starts, because:

- The honest occupancy-ceiling calculation will be reused.
- ADR-0002 sets the scope the next demo lives inside.
- The "locate your kernel" recipe is what the next demo *teaches by example*.

## Suggested skills

- `grill-with-docs` — when drafting ADR-0002 or revising CONTEXT.md, stress-test
  the plan against the existing domain language and update CONTEXT.md / ADRs
  inline as decisions crystallise.
- `improve-codebase-architecture` — useful before starting Phase 2 to scan the
  current notebook/worker split for refactor opportunities (especially the
  occupancy-ceiling math, which the next demo will inherit).
