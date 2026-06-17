"""Marimo notebook: GPU context-sharing demo (Mode A vs B vs C).

Phase 1: develops on a single GPU. Mode B (MPS) auto-disables where the daemon
is unavailable (e.g. WSL2). All other comparisons run anywhere.

Run with:  uv run marimo edit gpu_demo.py
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # GPU context-sharing scaling demo

    **The question this answers:** you have **N independent workers** that each launch
    GPU kernels, and **one GPU**. Is it worth making them *share* a single CUDA context
    — via MPS or CUDA streams — or should you just run them as separate processes? And
    at what point does sharing stop helping?

    Three ways to run N workers on one GPU:

    - **Mode A** — N processes, each with its **own** CUDA context. The driver
      time-slices the GPU between contexts: only one context's kernels run at a time,
      so spare capacity sits idle.
    - **Mode B** — N processes sharing **one** context via **MPS** (Multi-Process
      Service). Kernels from different processes can overlap on different SMs.
    - **Mode C** — **1 process**, one context, **N CUDA streams**. Same overlap as B,
      but no process boundary — a single host thread launches all streams.

    The sweep varies **occupancy** (how full the GPU is) and **kernel duration**; the
    heatmap shows, at each point, how much faster a shared mode (B or C) is than the
    no-sharing baseline (A).
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Key terms — click to expand": mo.md(r"""
- **Worker** — one unit of concurrent kernel submission: an OS *process* in Modes A/B,
  a CUDA *stream* in Mode C. Modes are always compared at constant N (worker count).
- **Occupancy** — how full the GPU's per-SM warp slots are. The heatmap x-axis is
  occupancy as **% of saturation** (100% = every SM packed). It's swept here via grid
  size (`blocks`), because block size and the kernel's resource use are held fixed.
- **Saturation** — the grid size at which *one* kernel alone fills the GPU
  (`SM count × blocks-per-SM`). Past it, a single kernel uses the whole device, so
  sharing can add nothing.
- **Launch-bound** — kernels so short the GPU finishes one faster than the host can
  launch the next; throughput is capped by the CPU's launch rate, not the GPU. The
  low-`work_per_thread` end of the y-axis.
- **Compute-bound** — kernels long enough that GPU execution dominates and the launch
  loop keeps up. The high-`work_per_thread` end.
- **Throughput** — `total kernels ÷ wall-clock` (kernels/sec); the median over trials.
- **Speedup** — a panel labelled *Mode X / Mode Y* is X's throughput ÷ Y's at the same
  (N, occupancy, duration) point.
"""),
        }
    )
    return


@app.cell
def _(mo):
    import cupy as cp
    import mps_helper
    import worker as worker_probe

    props = cp.cuda.runtime.getDeviceProperties(0)
    # These are all standard cudaDeviceProp fields. The CUDA struct spells them with
    # mixed casing (regsPer*M*ultiprocessor vs maxBlocksPer*M*ultiProcessor — not a typo
    # here, that inconsistency is in the headers); CuPy mirrors the names verbatim.
    # Fall back to Volta/Turing/Ampere/Hopper-common values if a CuPy build omits a field.
    max_threads_per_sm_detected = props.get("maxThreadsPerMultiProcessor", 2048)
    regs_per_sm = props.get("regsPerMultiprocessor", 65536)       # 64K regs/SM since Volta
    smem_per_sm = props.get("sharedMemPerMultiprocessor", 0)      # 0 = unknown → cap won't bind
    max_blocks_per_sm = props.get("maxBlocksPerMultiProcessor", 32)
    sm_count_detected = props["multiProcessorCount"]

    # Probe the demo kernel's real per-thread/per-block resource use. Accessing these
    # RawKernel attributes triggers an NVRTC compile and a cuFuncGetAttribute query, so
    # they reflect what nvcc actually allocated — not an estimate.
    probe_kernel = cp.RawKernel(worker_probe.KERNEL_SRC, "busy_kernel")
    kernel_regs = probe_kernel.num_regs
    kernel_smem = probe_kernel.shared_size_bytes

    device_info = {
        "name": props["name"].decode(),
        "compute_cap": f"{props['major']}.{props['minor']}",
        "sm_count": sm_count_detected,
        "max_threads_per_sm": max_threads_per_sm_detected,
        "mem_total_mib": props["totalGlobalMem"] // (1024 * 1024),
        "max_threads_per_block": props["maxThreadsPerBlock"],
    }
    mps_available = mps_helper.is_mps_available()

    banner = mo.md(
        f"""
    **GPU:** {device_info['name']} · CC {device_info['compute_cap']} ·
    **{device_info['sm_count']} SMs** · {device_info['max_threads_per_sm']} threads/SM ·
    {regs_per_sm // 1024}K regs/SM · {smem_per_sm // 1024} KiB smem/SM · {device_info['mem_total_mib']} MiB

    **Kernel `busy_kernel`:** {kernel_regs} regs/thread · {kernel_smem} bytes shared mem
    (light kernel — the resident-block ceiling is thread-limited, not register/smem-limited)

    **MPS daemon:** {'available — Mode B enabled' if mps_available else 'NOT available (likely WSL2) — Mode B disabled'}
    """
    )
    banner
    return (
        kernel_regs,
        kernel_smem,
        max_blocks_per_sm,
        max_threads_per_sm_detected,
        mps_available,
        mps_helper,
        regs_per_sm,
        smem_per_sm,
        sm_count_detected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Sweep configuration
    """)
    return


@app.cell
def _(mo, sm_count_detected):
    import os

    # SM count is editable so you can model a different GPU than the one you're on.
    sm_count_input = mo.ui.number(
        start=1,
        stop=2048,
        step=1,
        value=sm_count_detected,
        label="SM count (edit to model a different GPU)",
    )
    threads_per_block_sel = mo.ui.dropdown(
        options=["128", "256", "512"], value="256", label="Threads per block"
    )
    cpu_count = os.cpu_count() or 8

    mo.vstack(
        [
            mo.md("### Hardware model — block & worker options scale to this"),
            sm_count_input,
            threads_per_block_sel,
        ]
    )
    return cpu_count, sm_count_input, threads_per_block_sel


@app.cell
def _(
    cpu_count,
    kernel_regs,
    kernel_smem,
    max_blocks_per_sm,
    max_threads_per_sm_detected,
    mo,
    regs_per_sm,
    smem_per_sm,
    sm_count_input,
    threads_per_block_sel,
):
    threads_per_block = int(threads_per_block_sel.value)
    sm_count = int(sm_count_input.value)

    # Concurrent-block capacity per SM is the *smallest* of four hardware ceilings,
    # not just the thread count. A register- or shared-memory-heavy kernel hits one of
    # the resource caps first, so its grid saturates the GPU at a lower block count.
    # Resource caps are only included when their per-kernel input is known/positive
    # (busy_kernel uses 0 shared mem, so the shared-mem cap simply doesn't apply).
    occupancy_caps = {
        "thread": max_threads_per_sm_detected // threads_per_block,
        "block": max_blocks_per_sm,
    }
    if kernel_regs > 0:
        occupancy_caps["register"] = regs_per_sm // (threads_per_block * kernel_regs)
    if kernel_smem > 0 and smem_per_sm > 0:  # smem_per_sm == 0 means the device prop was unavailable
        occupancy_caps["shared-mem"] = smem_per_sm // kernel_smem

    blocks_per_sm = max(1, min(occupancy_caps.values()))
    binding_cap = min(occupancy_caps, key=occupancy_caps.get)
    saturation_blocks = sm_count * blocks_per_sm

    # Block options expressed as fractions/multiples of the saturation point,
    # so the occupancy axis means the same thing on any GPU.
    fractions = [0.05, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    block_options = sorted({max(1, round(f * saturation_blocks)) for f in fractions})
    block_option_strs = [str(b) for b in block_options]
    default_blocks = block_option_strs[::2] or [block_option_strs[0]]
    sat_caption = " · ".join(f"{b}={b / saturation_blocks * 100:.0f}%" for b in block_options)
    caps_caption = ", ".join(f"{name} {cap}" for name, cap in occupancy_caps.items())

    # Worker options: powers of two up to CPU count (Mode A/B are process-bound).
    worker_options = []
    _n = 1
    while _n <= cpu_count:
        worker_options.append(str(_n))
        _n *= 2
    default_workers = [w for w in ["1", "2", "4"] if w in worker_options] or [worker_options[0]]

    blocks_sel = mo.ui.multiselect(
        options=block_option_strs,
        value=default_blocks,
        label=f"Grid size (blocks) — saturation ≈ {saturation_blocks}",
    )
    n_workers_sel = mo.ui.multiselect(
        options=worker_options, value=default_workers, label="Worker counts (N)"
    )
    work_sel = mo.ui.multiselect(
        options=["100", "1000", "10000", "100000"],
        value=["1000", "10000"],
        label="work_per_thread — controls duration",
    )
    trials_sel = mo.ui.slider(1, 7, value=3, label="Trials per point (median is reported)")
    kernel_count_sel = mo.ui.slider(5, 50, value=20, label="Kernels per trial")
    enable_b = mo.ui.checkbox(value=False, label="Force-enable Mode B (only do this if MPS works)")

    mo.vstack(
        [
            mo.md(
                f"*Resident-block ceiling = {blocks_per_sm} blocks/SM, set by the "
                f"**{binding_cap} cap** (blocks/SM per cap: {caps_caption}).*"
            ),
            mo.md(f"*Block options (% of saturation):* {sat_caption}"),
            blocks_sel,
            n_workers_sel,
            work_sel,
            trials_sel,
            kernel_count_sel,
            enable_b,
        ]
    )
    return (
        binding_cap,
        blocks_per_sm,
        blocks_sel,
        enable_b,
        kernel_count_sel,
        n_workers_sel,
        saturation_blocks,
        threads_per_block,
        trials_sel,
        work_sel,
    )


@app.cell
def _(
    blocks_sel,
    enable_b,
    kernel_count_sel,
    mo,
    mps_available,
    n_workers_sel,
    saturation_blocks,
    trials_sel,
    work_sel,
):
    n_workers = sorted(int(x) for x in n_workers_sel.value)
    blocks_values = sorted(int(x) for x in blocks_sel.value)
    work_values = sorted(int(x) for x in work_sel.value)
    trials = trials_sel.value
    kernel_count = kernel_count_sel.value
    mode_b_enabled = mps_available and enable_b.value

    total_points = len(n_workers) * len(blocks_values) * len(work_values)
    modes_active = ["A", "C"] + (["B"] if mode_b_enabled else [])

    # Peak concurrent blocks the sweep will request, vs the GPU's saturation point.
    # Crossing ~100% is where B/C speedup over A should collapse toward 1.0.
    peak_concurrent = (max(n_workers) * max(blocks_values)) if n_workers and blocks_values else 0
    peak_pct = peak_concurrent / saturation_blocks * 100 if saturation_blocks else 0
    crosses_sat = "✅ crosses saturation" if peak_concurrent >= saturation_blocks else "⚠️ never saturates — raise blocks or N to see the collapse"

    mo.md(
        f"""
    **Sweep size:** {total_points} parameter points × {len(modes_active)} modes ({', '.join(modes_active)}) × {trials} trials
    = **{total_points * len(modes_active) * trials} runs**.

    Each run = warmup + {kernel_count} timed kernel launches.

    **Peak concurrent blocks:** {peak_concurrent} = **{peak_pct:.0f}%** of saturation ({saturation_blocks}) — {crosses_sat}
    """
    )
    return (
        blocks_values,
        kernel_count,
        mode_b_enabled,
        modes_active,
        n_workers,
        trials,
        work_values,
    )


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run sweep")
    run_button
    return (run_button,)


@app.cell
def _(
    blocks_values,
    kernel_count,
    mo,
    mode_b_enabled,
    modes_active,
    mps_helper,
    n_workers,
    run_button,
    threads_per_block,
    trials,
    work_values,
):
    import json
    import multiprocessing as mp
    import os as _os
    import statistics
    import time

    import worker as worker_mod

    _cache_path = _os.path.join("__marimo__", "sweep_cache.json")

    def _sweep():
        ctx = mp.get_context("spawn")

        def _run_processes(cfg, n, env_overlay=None):
            """Spawn n processes, each runs run_worker(cfg). Throughput = totalkernels / max(elapsed)."""
            init = worker_mod.set_env if env_overlay else None
            initargs = (env_overlay,) if env_overlay else ()
            with ctx.Pool(n, initializer=init, initargs=initargs) as pool:
                results = pool.map(worker_mod.run_worker, [cfg] * n)
            max_elapsed = max(r["elapsed_s"] for r in results)
            total_kernels = sum(r["kernel_count"] for r in results)
            return total_kernels / max_elapsed

        def _run_streams(cfg, n):
            r = worker_mod.run_streams(cfg, n)
            return r["throughput_kernels_per_s"]

        rows = []
        total_runs = len(n_workers) * len(blocks_values) * len(work_values) * len(modes_active) * trials

        sweep_start = time.perf_counter()

        if mode_b_enabled:
            mps_cm = mps_helper.mps_session()
            mps_env = mps_cm.__enter__()
        else:
            mps_cm = None
            mps_env = None

        try:
            with mo.status.progress_bar(total=total_runs, title="Sweeping") as progress:
                def _measure(mode, cfg, n, mps_env=None):
                    thr_trials = []
                    for _ in range(trials):
                        if mode == "C":
                            thr_trials.append(_run_streams(cfg, n))
                        else:
                            thr_trials.append(_run_processes(cfg, n, env_overlay=mps_env))
                        progress.update()
                    return statistics.median(thr_trials)

                for n in n_workers:
                    for blocks in blocks_values:
                        for work in work_values:
                            cfg = dict(
                                blocks=blocks,
                                threads_per_block=threads_per_block,
                                work_per_thread=work,
                                kernel_count=kernel_count,
                                warmup_count=5,
                            )
                            for mode in modes_active:
                                if mode == "B":
                                    thr = _measure(mode, cfg, n, mps_env=mps_env)
                                else:
                                    thr = _measure(mode, cfg, n)
                                rows.append(
                                    dict(
                                        mode=mode,
                                        n=n,
                                        blocks=blocks,
                                        work=work,
                                        throughput=thr,
                                    )
                                )
        finally:
            if mps_cm is not None:
                mps_cm.__exit__(None, None, None)

        return rows, time.perf_counter() - sweep_start

    if run_button.value:
        rows, sweep_elapsed = _sweep()
        # Persist so a kernel restart on the node (e.g. after the SSH session drops)
        # can reload results without re-running the sweep. __marimo__/ is gitignored.
        _os.makedirs("__marimo__", exist_ok=True)
        with open(_cache_path, "w") as _f:
            json.dump({"rows": rows, "sweep_elapsed": sweep_elapsed}, _f)
        from_cache = False
    elif _os.path.exists(_cache_path):
        with open(_cache_path) as _f:
            _cached = json.load(_f)
        rows = _cached["rows"]
        sweep_elapsed = _cached["sweep_elapsed"]
        from_cache = True
    else:
        mo.stop(True, mo.md("*Press **Run sweep** to start (no cached results found).*"))

    return from_cache, rows, sweep_elapsed


@app.cell
def _(from_cache, mo, rows, sweep_elapsed):
    import pandas as pd
    df = pd.DataFrame(rows)

    # Compute speedup vs Mode A at the same (n, blocks, work) point
    baseline = (
        df[df["mode"] == "A"]
        .set_index(["n", "blocks", "work"])["throughput"]
        .rename("baseline")
    )
    df = df.join(baseline, on=["n", "blocks", "work"])
    df["speedup_vs_A"] = df["throughput"] / df["baseline"]

    _header = (
        "### Results · loaded from cache (press *Run sweep* to refresh)"
        if from_cache
        else f"### Results · sweep took {sweep_elapsed:.1f}s"
    )
    mo.vstack(
        [
            mo.md(_header),
            mo.ui.table(df.round(3), pagination=True),
        ]
    )
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Regime heatmap
    """)
    return


@app.cell
def _(df, mo):
    # Derived from the loaded results (not the live config) so the view stays
    # consistent with whatever sweep is currently cached.
    _ns = sorted(df["n"].unique())
    n_for_heatmap = mo.ui.dropdown(
        options=[str(n) for n in _ns],
        value=str(_ns[-1]),
        label="N (worker count) shown on heatmap",
    )
    n_for_heatmap
    return (n_for_heatmap,)


@app.cell
def _(mo):
    scale_mode = mo.ui.radio(
        options=["auto-fit", "fixed"],
        value="auto-fit",
        label="Color range",
    )
    cmin_input = mo.ui.number(value=0.5, start=0.01, stop=100.0, step=0.05, label="cmin")
    cmax_input = mo.ui.number(value=2.0, start=0.01, stop=100.0, step=0.05, label="cmax")
    mo.hstack([scale_mode, cmin_input, cmax_input])
    return cmax_input, cmin_input, scale_mode


@app.cell
def _(cmax_input, cmin_input, df, n_for_heatmap, saturation_blocks, scale_mode):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    def _ratio(num, den, n):
        num_t = df[(df["mode"] == num) & (df["n"] == n)].pivot_table(
            index="work", columns="blocks", values="throughput"
        )
        den_t = df[(df["mode"] == den) & (df["n"] == n)].pivot_table(
            index="work", columns="blocks", values="throughput"
        )
        return num_t / den_t

    def _heatmap():
        n_sel = int(n_for_heatmap.value)

        # Which panels to render — Mode C/A is always present; the others appear when Mode B was swept.
        panels = [("C", "A")]
        if "B" in set(df["mode"]):
            panels.append(("B", "A"))
            panels.append(("C", "B"))

        pivots = [(num, den, _ratio(num, den, n_sel)) for num, den in panels]

        if scale_mode.value == "fixed":
            cmin = float(cmin_input.value)
            cmax = float(cmax_input.value)
        else:
            # Span every N (not just the selected one) so flipping the N dropdown
            # doesn't silently rebase the scale and break cross-N comparison.
            all_n = sorted(df["n"].unique())
            all_vals = np.concatenate(
                [_ratio(num, den, nn).values.flatten() for num, den in panels for nn in all_n]
            )
            finite = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
            if len(finite) > 0:
                # Symmetric in log-space so 1.0 stays at the colormap midpoint
                log_max = max(float(np.max(np.abs(np.log(finite)))), 0.05)
                cmax = float(np.exp(log_max))
                cmin = 1.0 / cmax
            else:
                cmin, cmax = 0.5, 2.0

        fig = make_subplots(
            rows=1,
            cols=len(pivots),
            subplot_titles=[f"Mode {num} / Mode {den} · N={n_sel}" for num, den, _ in pivots],
            shared_yaxes=True,
        )

        for i, (num, den, pivot) in enumerate(pivots, start=1):
            blocks_cols = list(pivot.columns)
            # x-axis reads as occupancy (% of saturation); the raw block count rides
            # along in customdata so hover still shows what was actually launched.
            occ_labels = [f"{b / saturation_blocks * 100:.0f}%" for b in blocks_cols]
            block_grid = np.tile(blocks_cols, (len(pivot.index), 1))
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=occ_labels,
                    y=[str(r) for r in pivot.index],
                    customdata=block_grid,
                    coloraxis="coloraxis",
                    hovertemplate=(
                        f"Mode {num} / Mode {den}<br>"
                        "occupancy=%{x} of saturation<br>"
                        "blocks=%{customdata}<br>"
                        "work_per_thread=%{y}<br>"
                        "speedup=%{z:.2f}x<extra></extra>"
                    ),
                ),
                row=1,
                col=i,
            )

        fig.update_xaxes(title="occupancy (% of GPU saturation)")
        fig.update_yaxes(
            title="work_per_thread (low → launch-bound · high → compute-bound)",
            col=1,
        )
        fig.update_layout(
            height=400,
            margin=dict(t=80, b=60),
            coloraxis=dict(
                colorscale="RdBu_r",
                cmin=cmin,
                cmax=cmax,
                cmid=1.0,
                colorbar=dict(title="speedup"),
            ),
        )
        return fig

    _heatmap()
    return (go,)


@app.cell
def _(mo):
    mo.md(r"""
    ### How to read the heatmap

    **Axes.** Columns = **occupancy** (% of saturation — 100% means one kernel already
    fills every SM). Rows = **kernel duration** (`work_per_thread`): the bottom is
    **launch-bound** (kernels so short the GPU waits on the CPU's launch rate), the top
    is **compute-bound** (GPU execution dominates). Hover any cell for the raw block count.

    **Cells.** Each panel is **Mode X / Mode Y** — X's throughput as a multiple of Y's
    at the same (N, occupancy, duration) point.

    - **Red** = X faster than Y (speedup > 1).
    - **Blue** = X slower than Y (speedup < 1).
    - **White** = roughly equal.

    **What to look for:**

    - **Low occupancy + longer kernels** (left, upper) — the sharing win. One kernel
      leaves the GPU mostly empty, so overlapping N of them on different SMs pays off:
      C/A and B/A go red.
    - **At / past saturation** (right columns) — the collapse. One kernel fills the GPU
      on its own, so sharing adds nothing and the panels fade to white/blue.
    - **Launch-bound corner** (bottom, worst at high N for Mode C) — Mode C funnels all
      N streams' launches through one host thread, while A/B launch from N processes in
      parallel, so **C can lose to A** here. C/B going blue as N grows is the same
      effect — a substrate gap, not a context-sharing one.

    When Mode B is active, three panels appear:

    - **C / A** — what *any* context-sharing buys over no sharing.
    - **B / A** — what MPS specifically buys over no sharing.
    - **C / B** — streams vs MPS, *given* both share a context. Mostly white = context
      sharing is the only lever; blue = MPS's parallel launching wins (e.g. launch-bound).

    Color scale: **auto-fit** spans every N in the sweep, so flipping the N selector
    keeps one comparable scale; **fixed** locks `cmin`/`cmax` for cross-sweep comparison.
    """)
    return


@app.cell
def _(
    binding_cap,
    blocks_per_sm,
    kernel_regs,
    kernel_smem,
    max_threads_per_sm_detected,
    mo,
    regs_per_sm,
    threads_per_block,
):
    _occ_pct = blocks_per_sm * threads_per_block / max_threads_per_sm_detected * 100
    mo.md(
        rf"""
    ### How to locate *your* kernel on the heatmap

    The heatmap is generic. To use it for a real kernel, map that kernel onto the two
    axes, then read the cell. Three steps + a sanity check.

    **1 · Find your column (occupancy).** Two compiled properties of a kernel decide how
    many of its thread-blocks fit on one SM at once — that ceiling sets where "saturation"
    is, and your grid size relative to it is your column:

    - *registers per thread* (`RawKernel.num_regs`) — each SM has a fixed register file
      ({regs_per_sm // 1024}K 32-bit registers here); more registers per thread → fewer
      resident threads.
    - *shared memory per block* (`RawKernel.shared_size_bytes`) — each SM has fixed
      shared memory; more per block → fewer resident blocks.

    The resident-block ceiling is the **smallest** of four caps (this is what the
    *Sweep configuration* banner computes for the active kernel):

    ```text
    blocks_per_SM = min(
        threads_per_SM // threads_per_block,                 # thread cap
        max_blocks_per_SM,                                   # hardware block cap
        regs_per_SM // (threads_per_block * regs_per_thread),# register cap
        smem_per_SM // smem_per_block,                       # shared-memory cap
    )
    saturation_blocks = SM_count * blocks_per_SM
    ```

    A kernel launched with `0.25 * saturation_blocks` sits in the 25% column.

    > **Worked example — this demo's `busy_kernel`:** {kernel_regs} regs/thread,
    > {kernel_smem} bytes shared mem. With these threads_per_block the **{binding_cap}
    > cap** binds → {blocks_per_sm} blocks/SM → **{_occ_pct:.0f}% theoretical occupancy**.
    > Because it's register/shared-mem-light, the thread cap is what binds, so the
    > "grid size = occupancy" shorthand holds exactly (see `CONTEXT.md`). A
    > register-heavy kernel would hit the *register* cap first, so the same grid size
    > would fill the GPU at a lower block count — its column would shift right.

    **2 · Find your row (duration regime).** Measure the kernel's steady-state GPU time:

    ```python
    from cupyx.profiler import benchmark
    r = benchmark(lambda: my_kernel((grid,), (block,), args), n_repeat=100)
    gpu_us = r.gpu_times.mean() * 1e6
    ```

    (or wrap launches in `cudaEventRecord` / `cudaEventElapsedTime`). Compare against
    host launch overhead, ~5–10 µs per launch:

    - GPU time **≪** launch overhead → **launch-bound** (bottom rows).
    - GPU time **≫** launch overhead → **compute-bound** (top rows).

    **3 · Read the cell.** With (column, row) in hand, read the **Mode C / Mode A**
    value: **>1 (red)** means sharing the GPU across N workers should speed your
    workload up; **≈1 (white)** means one kernel already fills the device and sharing
    buys nothing.

    **Sanity check — are you even in this heatmap's regime?** `busy_kernel` is pure
    arithmetic (compute-bound, no global-memory traffic). Before trusting a cell, confirm
    *your* kernel is too: estimate **arithmetic intensity** = FLOPs ÷ bytes moved from
    global memory. If it's low, your kernel can stall on HBM bandwidth — a *memory-bound*
    regime Phase 1 deliberately doesn't model (that's the Phase 2 demo). A quick tell:
    if `ncu`/`nsys` shows high `dram__throughput` with low SM utilisation, you're
    memory-bound and this heatmap won't predict your sharing behaviour.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Drilldown — throughput vs N

    Fix one (blocks, duration) point and watch throughput scale with N. A **flat** Mode C
    line is the launch-bound signature — more streams don't help when one host thread
    launches them all. A line that **rises then plateaus** is hitting saturation.
    """)
    return


@app.cell
def _(df, mo):
    _blocks = sorted(df["blocks"].unique())
    _works = sorted(df["work"].unique())
    drill_blocks = mo.ui.dropdown(
        options=[str(b) for b in _blocks],
        value=str(_blocks[len(_blocks) // 2]),
        label="blocks",
    )
    drill_work = mo.ui.dropdown(
        options=[str(w) for w in _works],
        value=str(_works[len(_works) // 2]),
        label="work_per_thread",
    )
    mo.hstack([drill_blocks, drill_work])
    return drill_blocks, drill_work


@app.cell
def _(df, drill_blocks, drill_work, go):
    def _drilldown():
        b = int(drill_blocks.value)
        w = int(drill_work.value)
        sub = df[(df["blocks"] == b) & (df["work"] == w)].sort_values(["mode", "n"])

        fig = go.Figure()
        for mode in sorted(sub["mode"].unique()):
            s = sub[sub["mode"] == mode]
            fig.add_trace(
                go.Scatter(
                    x=s["n"],
                    y=s["throughput"],
                    mode="lines+markers",
                    name=f"Mode {mode}",
                )
            )
        fig.update_layout(
            title=f"Throughput vs N · blocks={b}, work_per_thread={w}",
            xaxis_title="N (workers / streams)",
            yaxis_title="throughput (kernels/sec)",
            height=400,
        )
        return fig

    _drilldown()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
