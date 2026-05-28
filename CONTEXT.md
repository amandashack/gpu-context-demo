# GPU Context-Sharing Demo

A Marimo notebook that compares three ways for N independent workers to share one GPU
(separate contexts vs MPS-shared context vs streams in one process), and visualises
the regimes where each wins.

## Language

**Worker**:
One unit of concurrent kernel submission. In process-mode runs (Modes A, B) a worker is an OS process; in stream-mode runs (Mode C) a worker is a CUDA stream. Modes are compared at constant N (worker count); the substrate is what varies.
_Avoid_: "thread" (overloaded with CUDA threads), "task" (Marimo-overloaded), "process" alone (excludes Mode C)

**Mode**:
A point in the 2×2 of (process count) × (CUDA-context count). The three implemented modes are A, B, C; the fourth corner is documented but unimplemented (see below).

|                 | N contexts                | 1 context                       |
|-----------------|---------------------------|---------------------------------|
| **N processes** | **Mode A** — no sharing    | **Mode B** — MPS-shared        |
| **1 process**   | (Mode D, omitted)         | **Mode C** — N streams         |

- **Mode A** — N processes, each holding its own CUDA context on the GPU. Kernels from different contexts cannot be in flight simultaneously; the driver scheduler context-switches between them at kernel boundaries.
- **Mode B** — N processes, each connected as an MPS *client* to a single MPS *server* that owns one CUDA context. Kernels from different clients land in the same context and can overlap on different SMs.
- **Mode C** — 1 process holding 1 CUDA context with N non-blocking CUDA streams. Kernels in different streams can overlap on different SMs without any process boundary.
- **Mode D** *(omitted)* — 1 process, N OS threads, each thread holding its own CUDA context. Physically reachable via `cuCtxCreate`/`cuCtxSetCurrent` but not via CuPy's primary-context API. Deliberately omitted from Phase 1: it would decompose Mode A's overhead into process-boundary vs context-switch contributions, which is a *diagnostic* question better answered in Phase 2.

**Wall-clock**:
For a run, `max(elapsed_i)` across all Workers (process modes) or wall-clock time around all stream submissions (Mode C). Treats the run as one parallel job, finished when the slowest Worker is finished.
_Avoid_: "elapsed" alone (ambiguous between per-worker and aggregated)

**Throughput**:
`total_kernels ÷ wall_clock`, in kernels/sec. Measures the *launch-and-completion* rate of the run as a whole. Chosen instead of arithmetic throughput (GFLOPS) because the experiment is about scheduling overhead, not compute density — the kernel itself is identical across modes, so a compute-rate metric would barely move.
_Avoid_: "GFLOPS", "FLOPS/sec" (different metric, wrong question)

**Run**:
One execution of N Workers with a specific (Mode, n, blocks, work_per_thread) configuration. Includes warmups, the timed region, and a final device sync.

**Trial**:
One instance of a Run. The `trials` knob controls how many Trials per (Mode, n, blocks, work) point; the median Throughput across Trials is reported.

**Sweep**:
The full cross-product of all configured (n, blocks, work) values × all active Modes × `trials` Trials. Drives the heatmap and is what the "Run sweep" button triggers.

**Warmup**:
The 5 pre-timing kernel launches per Worker (per stream, in Mode C). Absorbs NVRTC JIT compilation, command-buffer allocation, instruction-cache filling, and other lazy-init costs so the timed region measures steady-state behavior.

**Occupancy / saturation**:
*Occupancy* is how full the GPU's per-SM warp slots are. *Saturation* is the grid size at which one kernel alone fills the device: `sm_count × (max_threads_per_sm ÷ threads_per_block)`. The heatmap x-axis is occupancy as **% of saturation**, and there is an exact identity behind the "blocks = occupancy" shorthand: with `threads_per_block` fixed and a resource-light kernel (so the per-SM ceiling is reachable), `blocks ÷ saturation_blocks` equals the device-average occupancy — the `threads_per_block` term cancels. Change either assumption (sweep block size, or use a register/shared-memory-heavy kernel) and grid size stops tracking occupancy; that fuller picture is Phase 2.
_Avoid_: implying occupancy depends on SM count — occupancy is per-SM; only the saturation *point* scales with SM count.

**Launch-bound / compute-bound**:
The two regimes along the duration axis. *Launch-bound*: kernels finish faster than the host can issue the next launch, so wall-clock is set by the CPU's launch rate and the GPU idles between kernels — the low-`work_per_thread` end. *Compute-bound*: GPU execution per kernel dominates and the launch loop keeps up — the high-`work_per_thread` end. The distinction is what makes Mode C's single launching thread a liability at high N in the launch-bound corner (Modes A/B launch from N processes in parallel), so C can lose to A there despite sharing a context.
_Avoid_: "memory-bound" — `busy_kernel` has no memory traffic to stall on; that's a Phase 2 / different-kernel regime.

## Relationships

- A run has exactly one **Mode** and one value of N (number of **Workers**).
- All Workers in a run execute the same kernel configuration.
- The taxonomy gives B and C a useful relationship: both test "N submission units sharing 1 context," differing only in substrate (MPS vs streams). Heatmap regions where B ≈ C are evidence that *context sharing* is the dominant lever; regions where B ≠ C are evidence that *substrate overhead* (IPC, JIT cache, per-process init) is showing through.

## Measurement methodology

Each Worker independently follows the pattern: **warmup → device sync → timed region → device sync**. Timing starts only after that Worker's own warmups complete, so every timed region measures steady-state.

- In **Mode C**, Warmups happen *per stream* (not per process), because each stream's command buffer is allocated lazily on its first launch — touching every stream during Warmup is what makes their command buffers ready for the timed region.
- In **Modes A and B**, NVRTC compiles the kernel **once per process** (N times total per Run). Mode C compiles once and reuses across all streams. The Warmups absorb this in both cases; the only externally-visible effect is that Mode A/B Runs take slightly longer to *start* producing results (longer setup before the timed region).
- The per-Worker timed windows are **not globally synchronized** across Workers in a Run. `pool.map` starts all N Workers ~simultaneously, but each one starts its own timer after its own warmups. Windows overlap by typically >95% of their duration; for small `kernel_count` or very large `work_per_thread` a slow Worker's warmup could leak into another Worker's timed region. Mitigation if it ever bites: add a `multiprocessing.Barrier` between warmup and timed regions.

## Audience

- **Phase 1** is aimed at a practitioner asking *"should I use MPS at all?"* — the heatmap's same-N baseline directly answers that.
- **Phase 2** shifts to practical implementation questions; a second view using **Mode A at N=1** as the baseline (scaling-efficiency lens) becomes useful there.

## Out of scope for Phase 1

- **`threads_per_block` is fixed at 256.** Affects warp count and register pressure, but is largely orthogonal to the context-sharing question. Sweeping it would multiply Sweep cost for marginal insight.
- **Single GPU only** — no multi-GPU and no multi-node. NVSHMEM and NCCL solve different problems (inter-GPU comms); see [ADR-0001](./docs/adr/0001-nvshmem-out-of-scope.md).
- **Compute-bound synthetic kernel only.** `busy_kernel` is pure arithmetic so the experiment isolates *scheduling overhead* from *bandwidth*. Memory-bound kernels are a separate question to be explored in Phase 2 (with H2D/D2H) or a different demo.
- **MPS tuning knobs left at defaults** — notably `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = 100%`. Constraining it would soft-partition the GPU among clients; that's a separate experiment about isolation/fairness, not about whether the sharing mechanism itself helps.

## Flagged for Phase 2

- An alternative aggregation — `sum(kernels) ÷ sum(elapsed)` instead of `÷ max(elapsed)` — would answer "how busy were the workers on average?" rather than "how long did the job take?" Useful for diagnosing whether pipelining is keeping all Workers busy uniformly or just shortening the critical path.
- A second heatmap with **Mode A at N=1** as baseline — shows scaling efficiency rather than "is sharing worth it?". Right view for the practical-implementation audience.
