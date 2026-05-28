# gpu-context-demo

An interactive [Marimo](https://marimo.io/) notebook that benchmarks three ways for N independent workers to share one GPU, and visualises where each one wins.

## The three modes

|                 | N CUDA contexts                              | 1 CUDA context                              |
|-----------------|----------------------------------------------|---------------------------------------------|
| **N processes** | **Mode A** — driver time-slices kernels       | **Mode B** — shared via MPS, kernels overlap |
| **1 process**   | (omitted, see `CONTEXT.md`)                  | **Mode C** — N CUDA streams                 |

The heatmap sweeps `(N workers, blocks-per-kernel, work-per-thread)` and reports throughput speedup over Mode A. The sharing win shows up at **low occupancy with kernels long enough not to be launch-bound**; it collapses once a single kernel saturates the GPU, and in the launch-bound corner Mode C can even lose to A (one host thread launching all streams vs N processes launching in parallel).

## Requirements

- Linux with an NVIDIA GPU and driver supporting CUDA 12.x (`nvidia-smi` to check)
- [uv](https://github.com/astral-sh/uv) for Python environment management
- Optional: `nvidia-cuda-mps-control` on PATH to enable Mode B (typically via `module load cuda/...` on HPC systems)

## Run it

```bash
uv sync
uv run marimo edit gpu_demo.py
```

For a remote GPU (e.g. tunneling from your laptop):

```bash
# On the GPU host:
uv run marimo edit --headless --port 8080 --no-token gpu_demo.py

# On your laptop:
ssh -L 8080:localhost:8080 <gpu-host>
# Open http://localhost:8080
```

Press **Run sweep** in the UI; tick **Force-enable Mode B** if MPS is available.

## Reading the heatmap

Columns are **occupancy** (% of GPU saturation); rows are **kernel duration** (`work_per_thread`) — low rows are launch-bound, high rows compute-bound. Each panel is *Mode X / Mode Y*:

- **Red** = X *faster* than Y at this `(N, occupancy, duration)` point
- **Blue** = X *slower* than Y
- **White** = no significant difference

The dropdown selects which N to view; the dropdowns below drill into throughput-vs-N for a fixed point. Hover any cell for the raw block count behind the occupancy %.

## Docs

- [`CONTEXT.md`](./CONTEXT.md) — glossary, methodology, what's deliberately out of scope
- [`docs/adr/`](./docs/adr/) — architecture decision records (currently: why NVSHMEM is out of scope)
