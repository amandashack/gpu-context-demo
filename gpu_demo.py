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

    Compares three ways for **N independent workers** to share **one GPU**:

    - **Mode A** — N processes, each with its own CUDA context. Driver time-slices kernels.
    - **Mode B** — N processes sharing one context via **MPS**. Kernels can overlap on the GPU.
    - **Mode C** — 1 process with **N CUDA streams**. No process boundary at all.

    Sweep occupancy (`blocks`) and per-kernel duration (`work_per_thread`); read off
    the heatmap where each mode wins.
    """)
    return


@app.cell
def _(mo):
    import cupy as cp
    import mps_helper

    props = cp.cuda.runtime.getDeviceProperties(0)
    # maxThreadsPerMultiProcessor is a standard cudaDeviceProp field; default to 2048
    # (the value for every compute capability since Kepler) if a CuPy build omits it.
    max_threads_per_sm_detected = props.get("maxThreadsPerMultiProcessor", 2048)
    sm_count_detected = props["multiProcessorCount"]
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
    **{device_info['sm_count']} SMs** · {device_info['max_threads_per_sm']} threads/SM · {device_info['mem_total_mib']} MiB

    **MPS daemon:** {'available — Mode B enabled' if mps_available else 'NOT available (likely WSL2) — Mode B disabled'}
    """
    )
    banner
    return max_threads_per_sm_detected, mps_available, mps_helper, sm_count_detected


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
def _(cpu_count, max_threads_per_sm_detected, mo, sm_count_input, threads_per_block_sel):
    threads_per_block = int(threads_per_block_sel.value)
    sm_count = int(sm_count_input.value)

    # Concurrent-block capacity at full occupancy. Thread-limited: valid for
    # threads_per_block >= 128, where the per-SM block cap (16-32) isn't binding.
    blocks_per_sm = max(1, max_threads_per_sm_detected // threads_per_block)
    saturation_blocks = sm_count * blocks_per_sm

    # Block options expressed as fractions/multiples of the saturation point,
    # so the occupancy axis means the same thing on any GPU.
    fractions = [0.05, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    block_options = sorted({max(1, round(f * saturation_blocks)) for f in fractions})
    block_option_strs = [str(b) for b in block_options]
    default_blocks = block_option_strs[::2] or [block_option_strs[0]]
    sat_caption = " · ".join(f"{b}={b / saturation_blocks * 100:.0f}%" for b in block_options)

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
def _(cmax_input, cmin_input, df, mo, n_for_heatmap, scale_mode):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    def _heatmap():
        n_sel = int(n_for_heatmap.value)

        # Which panels to render — Mode C/A is always present; the others appear when Mode B was swept.
        panels = [("C", "A")]
        if "B" in set(df["mode"]):
            panels.append(("B", "A"))
            panels.append(("C", "B"))

        def _ratio(num, den):
            num_t = df[(df["mode"] == num) & (df["n"] == n_sel)].pivot_table(
                index="work", columns="blocks", values="throughput"
            )
            den_t = df[(df["mode"] == den) & (df["n"] == n_sel)].pivot_table(
                index="work", columns="blocks", values="throughput"
            )
            return num_t / den_t

        pivots = [(num, den, _ratio(num, den)) for num, den in panels]

        if scale_mode.value == "fixed":
            cmin = float(cmin_input.value)
            cmax = float(cmax_input.value)
        else:
            all_vals = np.concatenate([p[2].values.flatten() for p in pivots])
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
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=[str(c) for c in pivot.columns],
                    y=[str(r) for r in pivot.index],
                    coloraxis="coloraxis",
                    hovertemplate=(
                        f"Mode {num} / Mode {den}<br>"
                        "blocks=%{x}<br>work=%{y}<br>"
                        "speedup=%{z:.2f}x<extra></extra>"
                    ),
                ),
                row=1,
                col=i,
            )

        fig.update_xaxes(title="blocks (occupancy)")
        fig.update_yaxes(title="work_per_thread (duration)", col=1)
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

    Each panel is **Mode X / Mode Y** — it shows X's throughput as a multiple of Y's
    at the same (N, blocks, work) point.

    - **Red** = X faster than Y (speedup > 1).
    - **Blue** = X slower than Y (speedup < 1).
    - **White** = roughly equal.

    When Mode B is active, three panels appear:

    - **C / A** — what *any* context-sharing buys you over no sharing.
    - **B / A** — what MPS specifically buys you over no sharing.
    - **C / B** — does the streams substrate beat the MPS substrate, given both
      share a context? Mostly white means context sharing is the only lever;
      mostly red means streams have less overhead than MPS; mostly blue means MPS
      schedules more aggressively somehow.

    All panels share one color scale. Toggle between **auto-fit** (uses the full
    dynamic range of this sweep — best contrast within a run) and **fixed** (lock
    `cmin`/`cmax` for cross-sweep comparability).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Drilldown — throughput vs N
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
