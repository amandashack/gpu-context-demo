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
    device_info = {
        "name": props["name"].decode(),
        "compute_cap": f"{props['major']}.{props['minor']}",
        "sm_count": props["multiProcessorCount"],
        "mem_total_mib": props["totalGlobalMem"] // (1024 * 1024),
        "max_threads_per_block": props["maxThreadsPerBlock"],
    }
    mps_available = mps_helper.is_mps_available()

    banner = mo.md(
        f"""
    **GPU:** {device_info['name']} · CC {device_info['compute_cap']} ·
    **{device_info['sm_count']} SMs** · {device_info['mem_total_mib']} MiB

    **MPS daemon:** {'available — Mode B enabled' if mps_available else 'NOT available (likely WSL2) — Mode B disabled'}
    """
    )
    banner
    return mps_available, mps_helper


@app.cell
def _(mo):
    mo.md(r"""
    ## Sweep configuration
    """)
    return


@app.cell
def _(mo):
    n_workers_sel = mo.ui.multiselect(
        options=["1", "2", "4", "8"], value=["1", "2", "4"], label="Worker counts (N)"
    )
    blocks_sel = mo.ui.multiselect(
        options=["1", "5", "10", "20", "40"],
        value=["1", "10", "40"],
        label="Grid size (blocks) — controls occupancy",
    )
    work_sel = mo.ui.multiselect(
        options=["100", "1000", "10000", "100000"],
        value=["1000", "10000"],
        label="work_per_thread — controls duration",
    )
    trials_sel = mo.ui.slider(1, 7, value=3, label="Trials per point (median is reported)")
    kernel_count_sel = mo.ui.slider(5, 50, value=20, label="Kernels per trial")
    enable_b = mo.ui.checkbox(value=False, label="Force-enable Mode B (only do this if MPS works)")

    mo.vstack([n_workers_sel, blocks_sel, work_sel, trials_sel, kernel_count_sel, enable_b])
    return (
        blocks_sel,
        enable_b,
        kernel_count_sel,
        n_workers_sel,
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

    mo.md(
        f"""
    **Sweep size:** {total_points} parameter points × {len(modes_active)} modes ({', '.join(modes_active)}) × {trials} trials
    = **{total_points * len(modes_active) * trials} runs**.

    Each run = warmup + {kernel_count} timed kernel launches.
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
    trials,
    work_values,
):
    import multiprocessing as mp
    import statistics
    import time

    import worker as worker_mod

    if not run_button.value:
        mo.stop(True, mo.md("*Press **Run sweep** to start.*"))

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
                                threads_per_block=256,
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

    rows, sweep_elapsed = _sweep()
    return rows, sweep_elapsed


@app.cell
def _(mo, rows, sweep_elapsed):
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

    mo.vstack(
        [
            mo.md(f"### Results · sweep took {sweep_elapsed:.1f}s"),
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
def _(mo, n_workers):
    n_for_heatmap = mo.ui.dropdown(
        options=[str(n) for n in n_workers],
        value=str(n_workers[-1]),
        label="N (worker count) shown on heatmap",
    )
    n_for_heatmap
    return (n_for_heatmap,)


@app.cell
def _(df, mo, modes_active, n_for_heatmap):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def _heatmap():
        n_sel = int(n_for_heatmap.value)
        non_baseline_modes = [m for m in modes_active if m != "A"]

        if not non_baseline_modes:
            mo.stop(True, mo.md("*Only Mode A is active — nothing to compare against.*"))

        fig = make_subplots(
            rows=1,
            cols=len(non_baseline_modes),
            subplot_titles=[f"Mode {m} speedup over A · N={n_sel}" for m in non_baseline_modes],
            shared_yaxes=True,
        )

        for i, mode in enumerate(non_baseline_modes, start=1):
            sub = df[(df["mode"] == mode) & (df["n"] == n_sel)]
            pivot = sub.pivot_table(index="work", columns="blocks", values="speedup_vs_A")
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=[str(c) for c in pivot.columns],
                    y=[str(r) for r in pivot.index],
                    colorscale="RdBu_r",
                    zmid=1.0,
                    colorbar=dict(title="speedup", x=1.0 + 0.1 * (i - 1)) if i == len(non_baseline_modes) else None,
                    hovertemplate="blocks=%{x}<br>work=%{y}<br>speedup=%{z:.2f}x<extra></extra>",
                ),
                row=1,
                col=i,
            )

        fig.update_xaxes(title="blocks (occupancy)")
        fig.update_yaxes(title="work_per_thread (duration)", col=1)
        fig.update_layout(height=400, margin=dict(t=80, b=60))
        return fig

    _heatmap()
    return (go,)


@app.cell
def _(mo):
    mo.md(r"""
    ### How to read the heatmap

    - **Red** = the mode is faster than Mode A (baseline). MPS or streams paid off.
    - **Blue** = the mode is *slower* than Mode A. Overhead exceeded the benefit.
    - **White** = no significant difference.

    The interesting cells are where the heatmap is decisively red — that's a regime
    where the mode is doing real work. Use the dropdown above to see how the picture
    changes as N grows.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Drilldown — throughput vs N
    """)
    return


@app.cell
def _(blocks_values, mo, work_values):
    drill_blocks = mo.ui.dropdown(
        options=[str(b) for b in blocks_values],
        value=str(blocks_values[len(blocks_values) // 2]),
        label="blocks",
    )
    drill_work = mo.ui.dropdown(
        options=[str(w) for w in work_values],
        value=str(work_values[len(work_values) // 2]),
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
