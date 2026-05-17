# gpu-context-demo

An interactive [Marimo](https://marimo.io/) notebook that benchmarks three ways for N independent workers to share one GPU, and visualises where each one wins.

## The three modes

|                 | N CUDA contexts                              | 1 CUDA context                              |
|-----------------|----------------------------------------------|---------------------------------------------|
| **N processes** | **Mode A** — driver time-slices kernels       | **Mode B** — shared via MPS, kernels overlap |
| **1 process**   | (omitted, see `CONTEXT.md`)                  | **Mode C** — N CUDA streams                 |

The heatmap sweeps `(N workers, blocks-per-kernel, work-per-thread)` and reports throughput speedup over Mode A. The interesting regimes are short kernels with low occupancy at high N — that's where context sharing pays off.

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

- **Red** = mode is *faster* than Mode A at this `(N, blocks, work)` point
- **Blue** = mode is *slower* than Mode A
- **White** = no significant difference

The dropdown selects which N to view; the dropdowns below the heatmap let you drill into throughput-vs-N for a fixed `(blocks, work)`.

## Docs

- [`CONTEXT.md`](./CONTEXT.md) — glossary, methodology, what's deliberately out of scope
- [`docs/adr/`](./docs/adr/) — architecture decision records (currently: why NVSHMEM is out of scope)
