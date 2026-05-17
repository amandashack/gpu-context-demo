"""Worker module spawned by multiprocessing for Mode A (own context) and Mode B (MPS).

Must be importable so multiprocessing.spawn can reload it in child processes.
Each spawned worker creates its own CuPy context the first time it touches the GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
import cupy as cp

KERNEL_SRC = r"""
extern "C" __global__
void busy_kernel(float *out, int work_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)tid * 0.01f + 1.0f;
    #pragma unroll 1
    for (int i = 0; i < work_per_thread; i++) {
        x = fmaf(x, 1.0001f, 1e-7f);
    }
    out[tid] = x;
}
"""


def set_env(env_dict: dict) -> None:
    """Pool initializer: apply env overlay (e.g. CUDA_MPS_PIPE_DIRECTORY) in the child.

    Must live at module level so multiprocessing.spawn can pickle it by qualified name.
    """
    import os
    os.environ.update(env_dict)


@dataclass
class WorkerConfig:
    blocks: int             # grid size — controls occupancy
    threads_per_block: int  # typically 256
    work_per_thread: int    # controls per-kernel duration
    kernel_count: int       # kernels to launch per worker, timed
    warmup_count: int       # kernels to launch before timing


def run_worker(config_dict: dict) -> dict:
    """Run kernel_count back-to-back launches in this process. Return timings.

    Called by multiprocessing.spawn in a child process. Each child gets its own
    CuPy context the first time it touches the GPU.
    """

    c = WorkerConfig(**config_dict)
    kernel = cp.RawKernel(KERNEL_SRC, "busy_kernel")
    out = cp.zeros(c.blocks * c.threads_per_block, dtype=cp.float32)

    for _ in range(c.warmup_count):
        kernel((c.blocks,), (c.threads_per_block,), (out, c.work_per_thread))
    cp.cuda.runtime.deviceSynchronize()

    start = time.perf_counter()
    for _ in range(c.kernel_count):
        kernel((c.blocks,), (c.threads_per_block,), (out, c.work_per_thread))
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - start

    return {
        "elapsed_s": elapsed,
        "kernel_count": c.kernel_count,
        "throughput_kernels_per_s": c.kernel_count / elapsed,
    }


def run_streams(config_dict: dict, n_streams: int) -> dict:
    """Mode C: single process, n_streams parallel CUDA streams.

    non_blocking=True is critical — without it, every stream syncs against the
    default stream and concurrency dies.
    """

    c = WorkerConfig(**config_dict)
    kernel = cp.RawKernel(KERNEL_SRC, "busy_kernel")

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]
    buffers = [cp.zeros(c.blocks * c.threads_per_block, dtype=cp.float32) for _ in range(n_streams)]

    for s, buf in zip(streams, buffers):
        with s:
            for _ in range(c.warmup_count):
                kernel((c.blocks,), (c.threads_per_block,), (buf, c.work_per_thread))
    for s in streams:
        s.synchronize()

    start = time.perf_counter()
    for s, buf in zip(streams, buffers):
        with s:
            for _ in range(c.kernel_count):
                kernel((c.blocks,), (c.threads_per_block,), (buf, c.work_per_thread))
    for s in streams:
        s.synchronize()
    elapsed = time.perf_counter() - start

    total_kernels = c.kernel_count * n_streams
    return {
        "elapsed_s": elapsed,
        "kernel_count": total_kernels,
        "throughput_kernels_per_s": total_kernels / elapsed,
    }
