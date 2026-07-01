"""Marimo notebook: GPU kernel regimes — tells vs traps (Phase 2).

The companion to `gpu_demo.py`. Phase 1 asks *should N workers share one GPU?*
using a deliberately compute-bound kernel. Phase 2 asks the question Phase 1
cannot: **which bound condition does my kernel actually hit, and which tuning
knob moves it — and when does the source-level guess lie?**

Six exhibits form a gallery of *tells* (source-level signals you can trust) and
*traps* (regimes only measurement reveals). The reactive roofline is the
centrepiece: change one input, watch the operating point move.

Runs on a CUDA GPU (needs `cupy`); with no GPU it renders the committed
`regime_cache.json` — a captured A100 run — so it stays a teaching artifact on
any laptop. All heavy logic lives in `regimes.py`.

Run with:  uv run marimo edit gpu_kernel_regimes.py
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
    # GPU kernel regimes — tells vs traps

    **The thesis:** you can *read a kernel's source* to form a hypothesis about which
    performance regime it sits in — but the source is a hypothesis, not a measurement.
    The **compile stats** (registers/thread, shared memory, register spills) plus the
    **occupancy ceiling** plus **one measurement** are what confirm or destroy that
    hypothesis.

    Each exhibit predicts a regime from the source (its static **arithmetic intensity**
    against the A100 roofline) and then measures the real operating point. Some
    predictions hold — those are **tells**. Some are **traps**: the source-level guess
    is wrong, and only physics catches the lie.

    - **Controls** (AI-dial, SAXPY) — the source prediction holds. Positive controls.
    - **Traps** (L2-reuse, register spill, uncoalescing) — the source predicts one
      regime; the measurement lands somewhere else.
    - **Tell** (shared memory) — you *chose* on-chip reuse, and you can read its cost
      directly in the occupancy ceiling.

    Start at the **roofline** below, flip the *reveal* switch off, and commit to a guess
    before you look.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Key terms — click to expand": mo.md(r"""
- **Regime** — the *bound condition* a kernel hits. Three matter here:
  **compute-bound** (SM functional units saturated), **HBM-bound** (device DRAM
  bandwidth saturated, ~1.5 TB/s on A100), and **latency-bound** (stalled on
  memory/instruction latency — saturates *neither* roof). Distinct from the Phase 1
  sharing *mode* (A/B/C); never conflate the two.
- **Arithmetic intensity (AI)** — FLOPs ÷ bytes moved from global memory. A kernel's
  x-position on the roofline. Below the ridge → HBM-bound; above → compute-bound.
- **Roofline** — throughput ceiling as a function of AI: `min(peak_FP32, peak_HBM × AI)`.
  The two straight lines (a sloped memory roof and a flat compute roof) meet at the
  **ridge point** (~12.5 FLOPs/byte on A100).
- **Apparent GB/s** — `source-counted bytes ÷ time`. A *derived* quantity, **not** a
  hardware counter. That's exactly what makes the traps work as inferences: if apparent
  GB/s exceeds the physical DRAM peak, the source's byte count must be wrong.
- **Occupancy ceiling** — resident thread-blocks per SM, the *smallest* of four caps:
  `min(thread, block, register, shared-mem)`. The **binding cap** is which one wins;
  it's the honest ceiling reused from the Phase 1 notebook.
- **Register spill** — an array indexed by a runtime value can't live in the (non-
  addressable) register file, so it's forced to **local memory** (`local_size_bytes > 0`),
  adding hidden memory traffic the source never advertises.
- **Coalescing** — whether consecutive threads touch consecutive addresses. A runtime
  property of the address pattern, invisible in the arithmetic.
"""),
        }
    )
    return


@app.cell
def _(mo):
    import regimes

    peaks = regimes.detect_peaks()
    gpu_present = not peaks.get("cached", False)
    _ridge = regimes.ridge(peaks)

    _src = (
        "live device"
        if gpu_present
        else "**cached A100 run** (no GPU here — showing `regime_cache.json`)"
    )
    banner = mo.md(
        f"""
    **GPU:** {peaks['name']} · CC {peaks['cc']} · **{peaks['sm']} SMs** ·
    {peaks['threads_per_sm']} threads/SM · {peaks['regs_per_sm'] // 1024}K regs/SM ·
    {peaks['smem_per_sm'] // 1024} KiB smem/SM

    **Peaks:** FP32 ~ {peaks['peak_fp32_gflops'] / 1e3:.1f} TFLOP/s ·
    HBM ~ {peaks['peak_hbm_gbps']:.0f} GB/s · ridge ~ {_ridge:.1f} FLOPs/byte

    *Source: {_src}.*
    """
    )
    banner
    return gpu_present, peaks, regimes


@app.cell
def _(gpu_present, mo):
    # Re-measuring only matters on a real GPU; off-GPU the button is inert and the
    # notebook renders from the committed cache.
    run_button = mo.ui.run_button(
        label="Re-measure on this GPU" if gpu_present else "Re-measure (needs a GPU)",
        disabled=not gpu_present,
    )
    run_button
    return (run_button,)


@app.cell
def _(gpu_present, peaks, regimes, run_button):
    # Live path writes back to regime_cache.json so a later off-GPU open replays this
    # machine's numbers; otherwise we read the committed capture.
    if gpu_present and run_button.value:
        measured = regimes.run_measurements(peaks)
        regimes.save_cache(peaks, measured)
        from_cache = False
    else:
        measured = regimes.load_cache()
        from_cache = True

    records = regimes.all_records(peaks, measured)
    rec_index = {(r["key"], r["iters"]): r for r in records}
    return from_cache, rec_index, records


@app.cell
def _(mo):
    mo.md(r"""
    ## The roofline — every exhibit, one plot

    Points sit at *(source AI, measured GFLOP/s)*. A point **on** the roof reached the
    ceiling its source predicted; a point **below** the roof fell short; a point **above**
    the HBM roof is physically impossible from DRAM — so it must be cache-served.
    """)
    return


@app.cell
def _(mo, records, regimes):
    # Exhibit selector drives the three-pane detail below; the roofline always shows all.
    _labels = {
        f"{e['num']} · {e['title']}  ({e['category']})": e["key"] for e in regimes.EXHIBITS
    }
    exhibit_sel = mo.ui.dropdown(
        options=_labels, value=next(iter(_labels)), label="Inspect exhibit"
    )
    # The AI dial: which iters point on Exhibit 1 to emphasise — the live "slide up the roof".
    iters_slider = mo.ui.slider(
        steps=regimes.AI_DIAL_ITERS, value=8, label="AI-dial iters (Exhibit 1)"
    )
    reveal = mo.ui.switch(value=True, label="reveal measurements")

    mo.hstack([exhibit_sel, iters_slider, reveal], justify="start", gap=2)
    return exhibit_sel, iters_slider, reveal


@app.cell
def _(exhibit_sel, iters_slider, peaks, reveal, records, regimes):
    import numpy as np
    import plotly.graph_objects as go

    _CAT_COLOR = {"control": "#2ca02c", "trap": "#d62728", "tell": "#ff7f0e"}
    _ridge = regimes.ridge(peaks)
    _pk_fl = peaks["peak_fp32_gflops"]
    _pk_bw = peaks["peak_hbm_gbps"]

    def _roofline():
        xs = np.logspace(-1, np.log10(300), 200)
        mem_roof = _pk_bw * xs                       # sloped DRAM ceiling
        fig = go.Figure()

        # The two roofs (drawn full-length so an "above the HBM roof" point reads clearly).
        fig.add_trace(go.Scatter(
            x=xs, y=mem_roof, mode="lines", name="HBM ceiling (DRAM)",
            line=dict(color="#888", dash="dash"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=xs, y=[_pk_fl] * len(xs), mode="lines", name="FP32 compute ceiling",
            line=dict(color="#555", dash="dash"), hoverinfo="skip"))
        fig.add_vline(x=_ridge, line=dict(color="#bbb", width=1, dash="dot"),
                      annotation_text=f"ridge {_ridge:.1f}", annotation_position="top")

        sel_key = exhibit_sel.value

        if not reveal.value:
            # Guess-first mode: show only where the source *predicts* each point would
            # sit — on the lower-envelope roofline at its AI. Commit before revealing.
            gx = [r["src_ai"] for r in records]
            gy = [min(_pk_fl, _pk_bw * r["src_ai"]) for r in records]
            fig.add_trace(go.Scatter(
                x=gx, y=gy, mode="markers", name="source prediction",
                marker=dict(symbol="circle-open", size=11, color="#333", line=dict(width=2)),
                hovertemplate="predicted on-roof<br>AI=%{x:.2f}<extra></extra>"))
            fig.add_annotation(x=0.5, y=0.06, xref="paper", yref="paper", showarrow=False,
                               text="reveal OFF — these are source guesses on the roof. "
                                    "Commit, then flip <b>reveal</b>.",
                               font=dict(color="#666"))
        else:
            # AI-dial as a connected "dial" so the sweep across the ridge reads as motion.
            dial = [r for r in records if r["key"] == "ai_dial"]
            if dial:
                fig.add_trace(go.Scatter(
                    x=[r["src_ai"] for r in dial], y=[r["gflops"] for r in dial],
                    mode="lines", name="AI dial", line=dict(color=_CAT_COLOR["control"], width=1),
                    hoverinfo="skip"))

            # One marker per exhibit, coloured by category; the register pair and the
            # dial's non-selected iters ride along as smaller points.
            for r in records:
                is_sel = r["key"] == sel_key
                is_dial_pick = r["key"] == "ai_dial" and r["iters"] == iters_slider.value
                emphasise = is_sel or is_dial_pick
                fig.add_trace(go.Scatter(
                    x=[r["src_ai"]], y=[r["gflops"]], mode="markers",
                    name=r["title"], showlegend=False,
                    marker=dict(
                        size=18 if emphasise else 10,
                        color=_CAT_COLOR[r["category"]],
                        symbol="star" if is_dial_pick else "circle",
                        line=dict(width=2 if emphasise else 0, color="#000")),
                    hovertemplate=(
                        f"<b>{r['title']}</b>"
                        + (f" (iters={r['iters']})" if r['iters'] else "")
                        + f"<br>source AI=%{{x:.2f}}<br>{r['gflops']:.0f} GFLOP/s "
                          f"({r['frac_fl'] * 100:.1f}% peak)"
                          f"<br>apparent {r['apparent_gbps']:.0f} GB/s "
                          f"({r['frac_bw'] * 100:.1f}% HBM)"
                          f"<br><b>{r['verdict']}</b><extra></extra>")))

        fig.update_xaxes(type="log", title="arithmetic intensity (FLOPs / byte, source-counted)",
                         range=[-1, np.log10(300)])
        fig.update_yaxes(type="log", title="achieved throughput (GFLOP/s)")
        fig.update_layout(height=460, margin=dict(t=40, b=60, r=20),
                          legend=dict(orientation="h", y=1.08, x=0))
        return fig

    _roofline()
    return (go,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Source → compile → measure

    Three panes for the selected exhibit: what the **source** says, what the **compiler**
    allocated, and where the kernel actually **landed**. The trap is visible as the gap
    between the left pane's claim and the right pane's verdict.
    """)
    return


@app.cell
def _(exhibit_sel, iters_slider, mo, rec_index, regimes):
    _ex = regimes.EXHIBIT_BY_KEY[exhibit_sel.value]
    _iters = iters_slider.value if _ex.get("swept") else None
    selected = rec_index.get((_ex["key"], _iters))

    left = mo.md(
        f"""**{_ex['num']} · {_ex['title']}** — _{_ex['category']}_

{_ex['hook']}
"""
        + (f"\n> ⚠️ {_ex['caveat']}\n" if _ex.get("caveat") else "")
        + f"\n```cpp\n{_ex['src'].strip()}\n```"
    )
    return left, selected


@app.cell
def _(go, selected):
    def _occ_bar():
        occ = selected["occ"]
        names = list(occ["caps"].keys())
        vals = [occ["caps"][k] for k in names]
        colors = ["#d62728" if k == occ["binding"] else "#adb5bd" for k in names]
        fig = go.Figure(go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=colors, text=vals, textposition="outside"))
        fig.update_layout(
            title=f"occupancy caps — binding: <b>{occ['binding']}</b> "
                  f"→ {occ['blocks_per_sm']} blocks/SM = {occ['occ_pct']:.0f}%",
            xaxis_title="blocks/SM this cap allows", height=240,
            margin=dict(t=40, b=40, l=60, r=40), showlegend=False)
        return fig

    _occ_bar()
    return


@app.cell
def _(mo, selected):
    r = selected
    _trap = "  ·  🎯 **TRAP** — source guess was wrong" if r["trap"] else ""
    middle = mo.md(
        f"""**Compile stats**

| stat | value |
|---|---|
| registers / thread | {r['num_regs']} |
| shared mem / block | {r['smem']} B |
| spill (local mem) | **{r['spill']} B**{' ⚠️' if r['spill'] else ''} |
"""
    )
    right = mo.md(
        f"""**Measured**

- source AI: **{r['src_ai']:.2f}** FLOPs/byte → predicted **{r['predicted']}**
- {r['gflops']:.0f} GFLOP/s (**{r['frac_fl'] * 100:.1f}%** of peak FP32)
- apparent {r['apparent_gbps']:.0f} GB/s (**{r['frac_bw'] * 100:.1f}%** of HBM)

### → {r['verdict']}{_trap}
"""
    )
    mo.hstack([middle, right], widths=[1, 1.4], gap=2)
    return


@app.cell
def _(left):
    left
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Every exhibit at a glance
    """)
    return


@app.cell
def _(from_cache, mo, records):
    import pandas as pd

    _rows = [
        {
            "#": r["num"],
            "exhibit": r["title"] + (f" · iters={r['iters']}" if r["iters"] else ""),
            "kind": r["category"],
            "src AI": round(r["src_ai"], 2),
            "predicted": r["predicted"],
            "GFLOP/s": round(r["gflops"]),
            "% peak": round(r["frac_fl"] * 100, 1),
            "app. GB/s": round(r["apparent_gbps"]),
            "% HBM": round(r["frac_bw"] * 100, 1),
            "regs": r["num_regs"],
            "smem": r["smem"],
            "spill": r["spill"],
            "occ %": round(r["occ"]["occ_pct"]),
            "bind": r["occ"]["binding"],
            "verdict": r["verdict"],
            "trap": "🎯" if r["trap"] else "",
        }
        for r in records
    ]
    _src = "loaded from cache" if from_cache else "fresh measurement"
    mo.vstack([
        mo.md(f"### Results · {_src}"),
        mo.ui.table(pd.DataFrame(_rows), pagination=False, selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reading the traps

    - **L2-reuse (3)** — apparent bandwidth exceeds the DRAM peak (>100% HBM). You cannot
      pull that from DRAM, so the source's byte count is a fiction: the 4 MB buffer lives
      in the 40 MB L2 and most "reads" never touch memory. Physics catches the lie.
    - **Register spill (4)** — a one-character change (`r[(j+1)]` → `r[(j+d)]` with `d`
      runtime-dependent) forces the array to local memory. Same source, same FLOP count,
      opposite outcome — and *only* `spill > 0` reveals why.
    - **Uncoalescing (5)** — same source shape as SAXPY, ~20× less bandwidth. Coalescing
      is a runtime address property, invisible in the arithmetic intensity entirely.
    - **Shared memory (6)** — the tell you *choose*: 32 KB/block flips the binding cap
      from thread to smem and drops occupancy to 62%. Read it in the occupancy bar.

    ## Honest limitations

    This gallery infers regimes; it does not read hardware counters. Worth stating plainly:

    - **"Apparent GB/s" is derived, not measured** (`source_bytes ÷ time`). That's what
      makes the traps work as *inferences* — but it also means exhibits **4, 5, and 6 all
      collapse to the same "latency-bound" verdict** despite three different causes (spill
      traffic, uncoalesced DRAM, smem-throttled occupancy). Real counters
      (`dram__bytes.sum`, `l2_tex__t_sectors`, gld/gst efficiency) via Nsight Compute would
      separate them. That verdict-resolution upgrade is deliberately *not* in this cut.
    - **Occupancy is theoretical, not achieved.** It's the static `min(caps)` ceiling — no
      tail effects, no register-allocation granularity. Exhibit 4's runtime-index kernel
      reports **100% occupancy and still craters**: a live proof that occupancy ≠ throughput.
    - **The 0.25 "latency-bound" threshold is a magic number.** A kernel at 26% of both
      roofs reads throughput-bound; at 24%, latency-bound. The cliff is a heuristic.
    - **Exhibit 1's compute ceiling (~35% of peak) is really FP-latency-bound** — the single
      dependent FMA chain has no ILP. Even the clean control predicts *which roof*, never
      *how close* you get. A truly throughput-bound kernel needs independent accumulators.
    - **No explicit L2 flush between kernels**, so the reuse exhibits can be mildly
      order-dependent on what ran before (benchmark warmup mitigates, doesn't eliminate).
    """)
    return


if __name__ == "__main__":
    app.run()
