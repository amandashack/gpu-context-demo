"""MPS daemon detection and control.

On WSL2, the MPS daemon is unavailable — is_mps_available() returns False.
On bare-metal Linux with the daemon installed, start_mps()/stop_mps() bracket
a run so spawned workers share a single CUDA context.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager


def is_mps_available() -> bool:
    """True if nvidia-cuda-mps-control is on PATH."""
    return shutil.which("nvidia-cuda-mps-control") is not None


@contextmanager
def mps_session():
    """Context manager: start MPS daemon, yield env vars to pass to child procs, stop daemon.

    Sets CUDA_MPS_PIPE_DIRECTORY and CUDA_MPS_LOG_DIRECTORY to per-session temp dirs
    so concurrent runs don't collide.
    """
    if not is_mps_available():
        raise RuntimeError("MPS daemon (nvidia-cuda-mps-control) not found on PATH")

    pipe_dir = tempfile.mkdtemp(prefix="mps_pipe_")
    log_dir = tempfile.mkdtemp(prefix="mps_log_")
    env = {
        "CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
        "CUDA_MPS_LOG_DIRECTORY": log_dir,
    }

    daemon_env = {**os.environ, **env}
    subprocess.run(
        ["nvidia-cuda-mps-control", "-d"],
        env=daemon_env,
        check=True,
        capture_output=True,
    )

    # The daemon backgrounds itself; give it a moment to come up
    import time
    time.sleep(0.5)

    # Yield child-env overlay so callers can subprocess.Popen(env={**os.environ, **mps_env})
    yield env

    # Tell the control daemon to quit
    subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        text=True,
        env=daemon_env,
        capture_output=True,
    )
    shutil.rmtree(pipe_dir, ignore_errors=True)
    shutil.rmtree(log_dir, ignore_errors=True)
