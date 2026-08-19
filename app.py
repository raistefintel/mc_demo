"""Streamlit app: stock NumPy RNG vs MKL-accelerated NumPy (mkl_random).

Two workloads:
  * European option pricing (default) — single-batch Gaussian draw with a
    Black-Scholes reference.
  * GBM path simulation — full price paths + VaR.

Runs execute sequentially — stock NumPy on the left panel, mkl_random on the
right — so each backend gets exclusive CPU time. A live ticker updates elapsed
seconds every ~50 ms while a worker thread crunches numbers in the background.
"""

from __future__ import annotations

import os
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from mc_kernel import (
    DEFAULT_MKL_BRNG,
    DEFAULT_NUMPY_BG,
    HAS_MKL_RANDOM,
    MKL_BRNGS,
    NUMPY_BIT_GENERATORS,
    OptionResult,
    SimResult,
    simulate_european_default,
    simulate_european_mkl,
    simulate_gbm_default,
    simulate_gbm_mkl,
    value_at_risk,
)

try:
    import mkl  # from `mkl-service` -- optional, enables runtime thread control
    HAS_MKL_SERVICE = True
except ImportError:
    HAS_MKL_SERVICE = False

st.set_page_config(page_title="Stock MC: NumPy vs Intel MKL", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.6rem; max-width: 1400px; }

/* Hero card ------------------------------------------------------------- */
.hero {
    background: linear-gradient(135deg, #0b1f3a 0%, #0068C9 100%);
    border-radius: 16px;
    padding: 1.6rem 2rem 1.7rem;
    margin: 0 0 1rem;
    color: #ffffff;
    box-shadow: 0 6px 24px rgba(0, 32, 80, 0.14);
}
.hero-eyebrow {
    font-size: .72rem; letter-spacing: .2em; text-transform: uppercase;
    opacity: .78; margin-bottom: .45rem; font-weight: 600;
}
.hero-title {
    font-size: 2.05rem; font-weight: 700; letter-spacing: -0.02em;
    line-height: 1.15; margin: 0;
}
.hero-tagline {
    margin-top: .65rem; font-size: 1.02rem; opacity: .92;
    line-height: 1.5; max-width: 920px;
}
.hero code {
    background: rgba(255, 255, 255, 0.16);
    padding: 1px 6px; border-radius: 4px; font-size: .92em;
}

/* Session scoreboard ---------------------------------------------------- */
.scoreboard {
    background: #0d1117; color: #e6edf3;
    border: 1px solid #21262d; border-radius: 14px;
    padding: 1.2rem 1.5rem 1.3rem;
    margin: .1rem 0 1.2rem;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.10);
}
.sb-head {
    display: flex; justify-content: space-between; align-items: center;
    font-size: .72rem; letter-spacing: .18em; text-transform: uppercase;
    color: #8b949e; margin-bottom: 1.05rem; font-weight: 600;
}
.sb-pill {
    background: #21262d; color: #e6edf3;
    padding: .22rem .7rem; border-radius: 999px;
    font-size: .66rem; letter-spacing: .14em; font-weight: 600;
}
.sb-idle {
    color: #8b949e; font-size: .95rem; line-height: 1.55;
    padding: .3rem 0 .1rem;
}
.sb-idle b { color: #e6edf3; }
.sb-row {
    display: flex; align-items: flex-end; gap: 2.4rem;
    flex-wrap: wrap; margin-bottom: 1.15rem;
}
.sb-hero-num {
    font-size: 3.4rem; font-weight: 800; color: #FF8C00;
    line-height: 1; letter-spacing: -0.03em;
}
.sb-hero-num .sb-x { font-size: 2rem; margin-left: .1rem; opacity: .9; }
.sb-hero-lbl {
    color: #8b949e; font-size: .7rem; letter-spacing: .14em;
    text-transform: uppercase; margin-top: .35rem; font-weight: 600;
}
.sb-stat-num {
    font-size: 1.7rem; font-weight: 700; color: #e6edf3;
    line-height: 1; letter-spacing: -0.02em;
}
.sb-stat-num .sb-unit {
    font-size: 1rem; color: #8b949e; margin-left: .18rem;
    font-weight: 500;
}
.sb-stat-lbl {
    color: #8b949e; font-size: .68rem; letter-spacing: .12em;
    text-transform: uppercase; margin-top: .35rem; font-weight: 600;
}
.sb-bar-row {
    display: flex; align-items: center; gap: .8rem;
    margin: .38rem 0; font-size: .82rem;
}
.sb-bar-label {
    width: 108px; color: #8b949e; text-align: right;
    text-transform: uppercase; letter-spacing: .1em;
    font-size: .68rem; font-weight: 600;
}
.sb-bar-track {
    flex: 1; background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; height: 18px; overflow: hidden;
}
.sb-bar-fill {
    display: block; height: 100%; border-radius: 5px;
    transition: width .4s ease;
}
.sb-bar-fill.numpy { background: linear-gradient(90deg, #0068C9, #4C9AFF); }
.sb-bar-fill.mkl   { background: linear-gradient(90deg, #FF8C00, #FFB347); }
.sb-bar-val {
    width: 74px; text-align: right; color: #e6edf3;
    font-weight: 600; font-variant-numeric: tabular-nums;
}

/* Panel headers --------------------------------------------------------- */
.panel-hdr {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: .3rem 0 .35rem .9rem;
    margin-bottom: .3rem;
}
.panel-hdr .p-title {
    font-size: .74rem; letter-spacing: .18em;
    text-transform: uppercase; color: #24292f; font-weight: 700;
}
.panel-hdr .p-sub {
    font-size: .8rem; color: #57606a; margin-top: .22rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.panel-hdr .p-status {
    display: flex; align-items: center; gap: .38rem;
    font-size: .66rem; letter-spacing: .18em;
    text-transform: uppercase; font-weight: 700;
}
.panel-hdr .p-dot { width: .55rem; height: .55rem; border-radius: 50%; }
.panel-timer {
    text-align: center;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 4.1rem; font-weight: 700;
    line-height: 1.1; letter-spacing: -.02em;
    margin: .35rem 0 .55rem;
    font-variant-numeric: tabular-nums;
}
</style>
""",
    unsafe_allow_html=True,
)


def _init_scoreboard() -> None:
    if "scoreboard" not in st.session_state:
        st.session_state.scoreboard = {
            "rounds": 0,
            "options_priced": 0,
            "notional": 0.0,
            "elapsed_numpy": 0.0,
            "elapsed_mkl": 0.0,
            "best_speedup": 0.0,
            "last_speedups": [],
        }
    if "race_pending" not in st.session_state:
        st.session_state.race_pending = False


_init_scoreboard()


def _fmt_dollars(x: float) -> str:
    if x >= 1e9:
        return f"${x/1e9:,.2f} B"
    if x >= 1e6:
        return f"${x/1e6:,.1f} M"
    if x >= 1e3:
        return f"${x/1e3:,.0f} K"
    return f"${x:,.0f}"


def _countdown(container, seconds: int = 3) -> None:
    """Big 3-2-1-GO flash before a race starts."""
    for i in range(seconds, 0, -1):
        container.markdown(
            f"<h1 style='text-align:center;font-size:8rem;margin:0;"
            f"color:#FF8C00;font-family:monospace'>{i}</h1>",
            unsafe_allow_html=True,
        )
        time.sleep(0.7)
    container.markdown(
        "<h1 style='text-align:center;font-size:8rem;margin:0;"
        "color:#00C853;font-family:monospace'>GO!</h1>",
        unsafe_allow_html=True,
    )
    time.sleep(0.4)
    container.empty()


def _render_scoreboard(placeholder) -> None:
    sb = st.session_state.scoreboard
    saved = sb["elapsed_numpy"] - sb["elapsed_mkl"]
    rolling = statistics.median(sb["last_speedups"]) if sb["last_speedups"] else 0.0
    best = sb["best_speedup"]

    if sb["rounds"] == 0:
        placeholder.markdown(
            """
<div class="scoreboard">
  <div class="sb-head">
    <span>Session scoreboard</span>
    <span class="sb-pill">Awaiting first race</span>
  </div>
  <div class="sb-idle">
    Hit <b>Start race</b> to run the head-to-head.
    Toggle <b>Continuous race</b> in the sidebar for hands-off kiosk mode —
    totals climb every round.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    total_max = max(sb["elapsed_numpy"], sb["elapsed_mkl"]) or 1.0
    np_pct = 100.0 * sb["elapsed_numpy"] / total_max
    mkl_pct = 100.0 * sb["elapsed_mkl"] / total_max

    opt_stat = ""
    if sb["options_priced"] > 0:
        opt_stat = (
            '<div>'
            f'<div class="sb-stat-num">{sb["options_priced"] / 1e6:.1f}'
            '<span class="sb-unit">M</span></div>'
            '<div class="sb-stat-lbl">options priced</div>'
            '</div>'
            '<div>'
            f'<div class="sb-stat-num">{_fmt_dollars(sb["notional"])}</div>'
            '<div class="sb-stat-lbl">notional</div>'
            '</div>'
        )

    rounds_lbl = "round" if sb["rounds"] == 1 else "rounds"
    placeholder.markdown(
        f"""
<div class="scoreboard">
  <div class="sb-head">
    <span>Session scoreboard</span>
    <span class="sb-pill">{sb['rounds']} {rounds_lbl}</span>
  </div>
  <div class="sb-row">
    <div>
      <div class="sb-hero-num">{rolling:.1f}<span class="sb-x">×</span></div>
      <div class="sb-hero-lbl">median MKL speedup</div>
    </div>
    <div>
      <div class="sb-stat-num">{saved:.1f}<span class="sb-unit">s</span></div>
      <div class="sb-stat-lbl">compute saved</div>
    </div>
    <div>
      <div class="sb-stat-num">{best:.1f}<span class="sb-unit">×</span></div>
      <div class="sb-stat-lbl">peak speedup</div>
    </div>
    {opt_stat}
  </div>
  <div class="sb-bar-row">
    <span class="sb-bar-label">stock&nbsp;NumPy</span>
    <span class="sb-bar-track">
      <span class="sb-bar-fill numpy" style="width:{np_pct:.1f}%"></span>
    </span>
    <span class="sb-bar-val">{sb['elapsed_numpy']:.1f} s</span>
  </div>
  <div class="sb-bar-row">
    <span class="sb-bar-label">MKL-accel</span>
    <span class="sb-bar-track">
      <span class="sb-bar-fill mkl" style="width:{mkl_pct:.1f}%"></span>
    </span>
    <span class="sb-bar-val">{sb['elapsed_mkl']:.1f} s</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# Attribute assignment on a plain dataclass is atomic under CPython's GIL, so
# no lock is needed for a UI ticker that tolerates slightly stale reads.
@dataclass
class Progress:
    step: int = 0
    total: int = 0
    elapsed: float = 0.0
    start_time: Optional[float] = None
    done: bool = False
    result: object = None
    error: Optional[str] = None

    def cb(self, step: int, total: int, elapsed: float) -> None:
        self.step, self.total, self.elapsed = step, total, elapsed


_STATUS_STYLE = {
    "idle":    ("#6a737d", "READY"),
    "running": ("#FFB000", "RUNNING"),
    "done":    ("#2ea043", "DONE"),
}


def render_panel(container, label, prog, n_paths, threads, status):
    title, subtitle, color = label
    live = status == "running" and prog.start_time is not None and not prog.done
    elapsed = (time.perf_counter() - prog.start_time) if live else prog.elapsed
    samples = prog.step * n_paths
    tput = samples / prog.elapsed if prog.elapsed > 0 else 0.0

    dot_color, status_txt = _STATUS_STYLE[status]
    timer_color = "#8b949e" if status == "idle" else color
    timer_txt = "—.—— s" if status == "idle" else f"{elapsed:6.2f} s"

    container.markdown(
        f"""
<div class="panel-hdr" style="border-left:4px solid {color};">
  <div>
    <div class="p-title">{title}</div>
    <div class="p-sub">{subtitle}</div>
  </div>
  <div class="p-status" style="color:{dot_color};">
    <span class="p-dot" style="background:{dot_color};"></span>{status_txt}
  </div>
</div>
<div class="panel-timer" style="color:{timer_color};">{timer_txt}</div>
""",
        unsafe_allow_html=True,
    )
    if status != "idle":
        container.progress(prog.step / prog.total if prog.total else 0.0)

    c1, c2, c3 = container.columns(3)
    c1.metric("Threads", threads)
    c2.metric("Samples drawn", "—" if status == "idle" else f"{samples/1e6:.1f} M")
    c3.metric("Throughput", "—" if status == "idle" else f"{tput/1e6:.1f} M/s")

    if prog.error:
        container.error(prog.error)
    elif status == "done":
        container.success(f"Completed in {prog.elapsed:.2f} s")


def run_phase(panel, sim_fn, kwargs, label, threads):
    """Run one backend in a thread, live-render its panel, return Progress."""
    prog = Progress()

    def worker():
        prog.start_time = time.perf_counter()
        try:
            prog.result = sim_fn(progress_cb=prog.cb, **kwargs)
        except Exception as exc:  # noqa: BLE001
            prog.error = f"{type(exc).__name__}: {exc}"
        prog.done = True

    threading.Thread(target=worker, daemon=True).start()
    while not prog.done:
        with panel.container():
            render_panel(st, label, prog, kwargs["n_paths"], threads, "running")
        time.sleep(0.05)
    with panel.container():
        render_panel(st, label, prog, kwargs["n_paths"], threads, "done")
    return prog


# ---------------- Sidebar ----------------
st.sidebar.header("Workload")
workload = st.sidebar.radio(
    "Benchmark",
    ["European option pricing", "GBM path simulation"],
    help=(
        "European option pricing draws n_paths Gaussians in one batch and "
        "prices a European call and put against the Black-Scholes closed form.\n\n"
        "GBM path simulation draws n_steps × n_paths Gaussians and produces "
        "full paths."
    ),
)
IS_OPTION = workload.startswith("European")

st.sidebar.header("Simulation parameters")
S0 = st.sidebar.number_input("Starting price S₀ ($)", 1.0, 10000.0, 100.0, step=10.0)

if IS_OPTION:
    K = st.sidebar.number_input("Strike K ($)", 1.0, 10000.0, 100.0, step=10.0)
    r = st.sidebar.slider("Risk-free rate r (annual)", 0.00, 0.15, 0.05, step=0.005)
    sigma = st.sidebar.slider("Volatility σ (annual)", 0.01, 1.00, 0.20, step=0.01)
    T = st.sidebar.slider("Time to expiry T (years)", 0.10, 5.0, 1.0, step=0.10)
    mu = None
    n_steps = None
    n_paths_choices = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
    n_paths_default = 10_000_000
else:
    mu = st.sidebar.slider("Expected return μ (annual)", -0.20, 0.30, 0.08, step=0.01)
    sigma = st.sidebar.slider("Volatility σ (annual)", 0.01, 1.00, 0.25, step=0.01)
    T = st.sidebar.slider("Time horizon T (years)", 0.25, 5.0, 1.0, step=0.25)
    n_steps = st.sidebar.select_slider("Time steps", [63, 126, 252, 504, 1008], 252)
    K = None
    r = None
    n_paths_choices = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
    n_paths_default = 1_000_000

n_paths = st.sidebar.select_slider("Monte Carlo paths", n_paths_choices, n_paths_default)
seed = st.sidebar.number_input("Random seed", 0, 2**31 - 1, 42)

st.sidebar.divider()
st.sidebar.subheader("RNG algorithm")
numpy_bg_names = list(NUMPY_BIT_GENERATORS)
numpy_bg = st.sidebar.selectbox(
    "NumPy BitGenerator",
    numpy_bg_names,
    index=numpy_bg_names.index(DEFAULT_NUMPY_BG),
    help="Algorithm used by numpy.random.Generator on the left panel.",
)
mkl_brng = st.sidebar.selectbox(
    "Intel oneMKL BRNG",
    MKL_BRNGS,
    index=MKL_BRNGS.index(DEFAULT_MKL_BRNG),
    help="VSL Basic Random Number Generator used by mkl_random on the right panel.",
    disabled=not HAS_MKL_RANDOM,
)
if numpy_bg == mkl_brng == "MT19937":
    st.sidebar.caption("✓ Both sides using identical MT19937 — fair head-to-head.")
elif numpy_bg != mkl_brng:
    st.sidebar.caption("⚠ Different engines selected — not a strict apples-to-apples race.")

st.sidebar.divider()
st.sidebar.subheader("Threading")
cpu_count = os.cpu_count() or 1
if HAS_MKL_SERVICE:
    mkl_threads = st.sidebar.slider("MKL thread count", 1, cpu_count, mkl.get_max_threads())
    mkl.set_num_threads(int(mkl_threads))
else:
    env = os.environ.get("MKL_NUM_THREADS", "")
    mkl_threads = int(env) if env.isdigit() else cpu_count
    st.sidebar.caption(f"Install `mkl-service` for live thread control (using {mkl_threads}).")

st.sidebar.divider()
st.sidebar.subheader("Booth mode")
continuous = st.sidebar.checkbox(
    "Continuous race",
    value=False,
    help="Auto-repeat rounds so the scoreboard totals climb continuously.",
)
show_countdown_ui = st.sidebar.checkbox("Show 3-2-1 countdown", value=True)
celebrate = st.sidebar.checkbox("Balloons on new best speedup", value=True)
if st.sidebar.button("Reset scoreboard"):
    if "scoreboard" in st.session_state:
        del st.session_state["scoreboard"]
    _init_scoreboard()
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    f"mkl_random: {'✓' if HAS_MKL_RANDOM else '✗'} · "
    f"mkl-service: {'✓' if HAS_MKL_SERVICE else '✗'} · "
    f"CPU cores: {cpu_count} · NumPy: {np.__version__}"
)

# ---------------- Main ----------------
NUMPY_LABEL = (
    "Stock NumPy",
    f"numpy.random.Generator(np.random.{numpy_bg}(seed))",
    "#0068C9",
)
MKL_LABEL = (
    "MKL-accelerated NumPy",
    f"mkl_random.RandomState(seed, brng='{mkl_brng}')",
    "#FF8C00",
)

if IS_OPTION:
    hero_title = (
        f"Drop-in speedup: pricing {int(n_paths):,} options with "
        f"MKL-accelerated NumPy"
    )
else:
    hero_title = (
        f"Drop-in speedup: {int(n_paths):,} stock paths with "
        f"MKL-accelerated NumPy"
    )

st.markdown(
    f"""
<div class="hero">
  <div class="hero-eyebrow">NumPy · same code · Intel oneMKL under the hood</div>
  <div class="hero-title">{hero_title}</div>
  <div class="hero-tagline">
    Same script. Same seed. Same CPU. Swap <code>numpy.random</code> for
    <code>mkl_random</code> — an <code>numpy.random.RandomState</code>-compatible
    drop-in — and the Gaussian sampler routes through Intel oneMKL VSL.
    No algorithm changes. Out-of-the-box speedup.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

scoreboard_placeholder = st.empty()
_render_scoreboard(scoreboard_placeholder)

start = st.button("▶ Start race", type="primary", use_container_width=True)
if start:
    st.session_state.race_pending = True

col_left, col_right = st.columns(2)
panel_left, panel_right = col_left.empty(), col_right.empty()
countdown_placeholder = st.empty()
result_area = st.container()

if not HAS_MKL_RANDOM:
    st.warning("`mkl_random` not installed — MKL side will error. `pip install mkl-random`")

# Initial idle render (persists in the placeholder until overwritten).
idle = Progress()
with panel_left.container():
    render_panel(st, NUMPY_LABEL, idle, int(n_paths), 1, "idle")
with panel_right.container():
    render_panel(st, MKL_LABEL, idle, int(n_paths), mkl_threads, "idle")

if st.session_state.race_pending and IS_OPTION:
    st.session_state.race_pending = False
    is_looping = st.session_state.pop("is_looping", False)
    if show_countdown_ui and not is_looping:
        _countdown(countdown_placeholder, 3)
    kwargs = dict(
        S0=float(S0), K=float(K), r=float(r), sigma=float(sigma), T=float(T),
        n_paths=int(n_paths), seed=int(seed),
    )
    prog_num = run_phase(
        panel_left, simulate_european_default,
        {**kwargs, "bit_generator": numpy_bg}, NUMPY_LABEL, 1,
    )
    if HAS_MKL_RANDOM:
        prog_mkl = run_phase(
            panel_right, simulate_european_mkl,
            {**kwargs, "brng": mkl_brng}, MKL_LABEL, mkl_threads,
        )
    else:
        prog_mkl = Progress(error="mkl_random not installed", done=True)
        with panel_right.container():
            render_panel(st, MKL_LABEL, prog_mkl, int(n_paths), mkl_threads, "done")

    r_num, r_mkl = prog_num.result, prog_mkl.result
    if r_num and r_mkl:
        speedup = r_num.elapsed_s / r_mkl.elapsed_s
        sb = st.session_state.scoreboard
        sb["rounds"] += 1
        sb["options_priced"] += r_mkl.n_paths
        sb["notional"] += r_mkl.call_price * r_mkl.n_paths
        sb["elapsed_numpy"] += r_num.elapsed_s
        sb["elapsed_mkl"] += r_mkl.elapsed_s
        sb["last_speedups"] = (sb["last_speedups"] + [speedup])[-5:]
        new_best = speedup > sb["best_speedup"]
        if new_best:
            sb["best_speedup"] = speedup
        _render_scoreboard(scoreboard_placeholder)
        result_area.markdown("## The race")

        c1, c2, c3 = result_area.columns(3)
        c1.metric(f"NumPy ({numpy_bg})", f"{r_num.elapsed_s:.3f} s",
                  f"{r_num.options_per_s/1e6:.1f} M options/s")
        c2.metric(f"Intel oneMKL ({mkl_brng})", f"{r_mkl.elapsed_s:.3f} s",
                  f"{r_mkl.options_per_s/1e6:.1f} M options/s")
        c3.metric("Speedup", f"{speedup:.1f}×", "MKL vs NumPy")

        result_area.markdown("### Do both get the same answer?")
        call_bs, put_bs = r_num.call_bs, r_num.put_bs
        # Loose MC tolerance ~5σ over sqrt(n) — comfortably passes for large n.
        tol = 5.0 / float(np.sqrt(int(n_paths)))

        b1, b2, b3 = result_area.columns(3)
        b1.metric("Black-Scholes CALL", f"${call_bs:,.4f}")
        b2.metric(f"MC CALL ({numpy_bg})", f"${r_num.call_price:,.4f}",
                  f"err {r_num.call_error:.4f}")
        b3.metric(f"MC CALL ({mkl_brng})", f"${r_mkl.call_price:,.4f}",
                  f"err {r_mkl.call_error:.4f}")

        p1, p2, p3 = result_area.columns(3)
        p1.metric("Black-Scholes PUT", f"${put_bs:,.4f}")
        p2.metric(f"MC PUT ({numpy_bg})", f"${r_num.put_price:,.4f}",
                  f"err {r_num.put_error:.4f}")
        p3.metric(f"MC PUT ({mkl_brng})", f"${r_mkl.put_price:,.4f}",
                  f"err {r_mkl.put_error:.4f}")

        L1_num, L1_mkl = r_num.l1_error, r_mkl.l1_error
        if L1_num <= tol and L1_mkl <= tol:
            result_area.success(
                f"✓ TEST PASSED — NumPy L1 = {L1_num:.4f}, "
                f"MKL L1 = {L1_mkl:.4f}, tolerance = {tol:.4f}"
            )
        else:
            result_area.warning(
                f"NumPy L1 = {L1_num:.4f}, MKL L1 = {L1_mkl:.4f}, "
                f"tolerance = {tol:.4f} (raise n_paths to tighten)"
            )

        result_area.markdown("### Do both RNGs sample the same distribution?")
        result_area.caption(
            "Both RNGs draw N(0,1) samples. Matching moments here (and the "
            "overlay chart below) prove they're mathematically equivalent."
        )
        stats_table = {
            "Statistic": ["mean (target 0)", "std (target 1)", "min", "max", "skew (target 0)"],
            f"NumPy {numpy_bg}": [
                f"{r_num.stats['mean']:+.5f}",
                f"{r_num.stats['std']:.5f}",
                f"{r_num.stats['min']:.3f}",
                f"{r_num.stats['max']:.3f}",
                f"{r_num.stats['skew']:+.5f}",
            ],
            f"MKL {mkl_brng}": [
                f"{r_mkl.stats['mean']:+.5f}",
                f"{r_mkl.stats['std']:.5f}",
                f"{r_mkl.stats['min']:.3f}",
                f"{r_mkl.stats['max']:.3f}",
                f"{r_mkl.stats['skew']:+.5f}",
            ],
        }
        result_area.table(stats_table)

        result_area.markdown("### At a glance")
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        labels = [f"NumPy\n({numpy_bg})", f"Intel oneMKL\n({mkl_brng})"]
        colors = ["#0068C9", "#FF8C00"]

        times = [r_num.elapsed_s, r_mkl.elapsed_s]
        ax[0, 0].barh(labels, times, color=colors, height=0.55)
        for i, t in enumerate(times):
            ax[0, 0].text(t, i, f"  {t:.3f} s", va="center",
                          fontsize=12, fontweight="bold")
        ax[0, 0].set_title(f"Wall-clock — MKL finishes {speedup:.1f}× sooner",
                           fontsize=13, fontweight="bold")
        ax[0, 0].set_xlabel("seconds")
        ax[0, 0].invert_yaxis()
        ax[0, 0].grid(axis="x", alpha=0.3)

        tputs = [r_num.options_per_s / 1e6, r_mkl.options_per_s / 1e6]
        ax[0, 1].bar(labels, tputs, color=colors, width=0.55)
        for i, v in enumerate(tputs):
            ax[0, 1].text(i, v, f"{v:.1f}", ha="center", va="bottom",
                          fontsize=12, fontweight="bold")
        ax[0, 1].set_title("Throughput — million options / second",
                           fontsize=13, fontweight="bold")
        ax[0, 1].set_ylabel("M opt/s")
        ax[0, 1].grid(axis="y", alpha=0.3)

        Z_num, Z_mkl = r_num.z_sample, r_mkl.z_sample
        if Z_num.size and Z_mkl.size:
            disc = float(np.exp(-r * T))
            drift = (r - 0.5 * sigma * sigma) * T
            sig_sqrtT = sigma * float(np.sqrt(T))

            def _running_call(Z):
                S_T = S0 * np.exp(drift + sig_sqrtT * Z)
                pay = disc * np.maximum(S_T - K, 0.0)
                return np.cumsum(pay) / np.arange(1, len(pay) + 1)

            x = np.arange(1, len(Z_num) + 1)
            ax[1, 0].plot(x, _running_call(Z_num), color=colors[0],
                          lw=1.5, label=f"NumPy ({numpy_bg})")
            ax[1, 0].plot(x, _running_call(Z_mkl), color=colors[1],
                          lw=1.5, label=f"MKL ({mkl_brng})")
            ax[1, 0].axhline(call_bs, color="black", ls="--", lw=1.2,
                             label=f"Black-Scholes = ${call_bs:.3f}")
            ax[1, 0].set_xscale("log")
            ax[1, 0].set_title(
                f"MC CALL convergence — first {len(Z_num):,} of "
                f"{r_mkl.n_paths:,} paths",
                fontsize=13, fontweight="bold",
            )
            ax[1, 0].set_xlabel("paths (log scale)")
            ax[1, 0].set_ylabel("running MC estimate ($)")
            ax[1, 0].legend(loc="best", fontsize=9)
            ax[1, 0].grid(alpha=0.3)

            bins = np.linspace(-4.5, 4.5, 80)
            ax[1, 1].hist(Z_num, bins=bins, color=colors[0], alpha=0.55,
                          density=True, label=f"NumPy ({numpy_bg})")
            ax[1, 1].hist(Z_mkl, bins=bins, color=colors[1], alpha=0.55,
                          density=True, label=f"MKL ({mkl_brng})")
            xs = np.linspace(-4.5, 4.5, 200)
            pdf = np.exp(-0.5 * xs * xs) / np.sqrt(2 * np.pi)
            ax[1, 1].plot(xs, pdf, color="black", ls="--", lw=1.2,
                          label="N(0,1)")
            ax[1, 1].set_title(
                f"Gaussian samples overlay — first {len(Z_num):,} draws",
                fontsize=13, fontweight="bold",
            )
            ax[1, 1].set_xlabel("Z")
            ax[1, 1].set_ylabel("density")
            ax[1, 1].legend(loc="upper left", fontsize=9)
            ax[1, 1].grid(alpha=0.3)

        fig.tight_layout()
        result_area.pyplot(fig)

        if celebrate and new_best and speedup > 3.0:
            st.balloons()
        if continuous:
            st.session_state.race_pending = True
            st.session_state.is_looping = True
            time.sleep(1.2)
            st.rerun()
    elif r_num:
        result_area.error("MKL run did not complete. Install `mkl_random`.")
        result_area.metric(f"NumPy ({numpy_bg})", f"{r_num.elapsed_s:.3f} s")

elif st.session_state.race_pending:
    st.session_state.race_pending = False
    is_looping = st.session_state.pop("is_looping", False)
    if show_countdown_ui and not is_looping:
        _countdown(countdown_placeholder, 3)
    kwargs = dict(
        S0=float(S0), mu=float(mu), sigma=float(sigma),
        T=float(T), n_steps=int(n_steps), n_paths=int(n_paths), seed=int(seed),
    )
    prog_num = run_phase(
        panel_left, simulate_gbm_default,
        {**kwargs, "bit_generator": numpy_bg}, NUMPY_LABEL, 1,
    )
    if HAS_MKL_RANDOM:
        prog_mkl = run_phase(
            panel_right, simulate_gbm_mkl,
            {**kwargs, "brng": mkl_brng}, MKL_LABEL, mkl_threads,
        )
    else:
        prog_mkl = Progress(error="mkl_random not installed", done=True)
        with panel_right.container():
            render_panel(st, MKL_LABEL, prog_mkl, int(n_paths), mkl_threads, "done")

    r_num, r_mkl = prog_num.result, prog_mkl.result
    if r_num and r_mkl:
        speedup = r_num.elapsed_s / r_mkl.elapsed_s
        sb = st.session_state.scoreboard
        sb["rounds"] += 1
        sb["elapsed_numpy"] += r_num.elapsed_s
        sb["elapsed_mkl"] += r_mkl.elapsed_s
        sb["last_speedups"] = (sb["last_speedups"] + [speedup])[-5:]
        new_best = speedup > sb["best_speedup"]
        if new_best:
            sb["best_speedup"] = speedup
        _render_scoreboard(scoreboard_placeholder)
        result_area.markdown("## The race")
        c1, c2, c3 = result_area.columns(3)
        c1.metric(f"NumPy ({numpy_bg})", f"{r_num.elapsed_s:.2f} s",
                  f"{r_num.throughput_samples_per_s/1e6:.1f} M samples/s")
        c2.metric(f"Intel oneMKL ({mkl_brng})", f"{r_mkl.elapsed_s:.2f} s",
                  f"{r_mkl.throughput_samples_per_s/1e6:.1f} M samples/s")
        c3.metric("Speedup", f"{speedup:.1f}×", "MKL vs NumPy")

        final = r_mkl.paths[-1]
        var95 = value_at_risk(final, float(S0), 0.95)
        var99 = value_at_risk(final, float(S0), 0.99)
        result_area.markdown("### Financial output (from MKL run)")
        f1, f2, f3, f4 = result_area.columns(4)
        f1.metric("Mean final price", f"${final.mean():,.2f}")
        f2.metric("P(loss)", f"{float(np.mean(final < S0)) * 100:.1f}%")
        f3.metric("95% VaR", f"${var95:,.2f}")
        f4.metric("99% VaR", f"${var99:,.2f}")

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        idx = np.random.default_rng(0).choice(
            r_mkl.paths.shape[1], min(200, r_mkl.paths.shape[1]), replace=False
        )
        ax[0].plot(r_mkl.paths[:, idx], lw=0.5, alpha=0.4, color="tab:orange")
        ax[0].axhline(S0, color="black", ls="--", lw=1, label=f"S₀ = ${S0:.0f}")
        ax[0].set(title=f"200 sample paths (of {n_paths:,})",
                  xlabel="Time step", ylabel="Price ($)")
        ax[0].legend()
        ax[1].hist(final, bins=80, color="tab:orange", alpha=0.75)
        ax[1].axvline(S0, color="black", ls="--", lw=1, label=f"S₀ = ${S0:.0f}")
        ax[1].axvline(S0 - var95, color="red", ls=":", lw=1.5, label="95% VaR")
        ax[1].set(title="Distribution of final prices",
                  xlabel="Final price ($)", ylabel="Count")
        ax[1].legend()
        fig.tight_layout()
        result_area.pyplot(fig)

        if celebrate and new_best and speedup > 3.0:
            st.balloons()
        if continuous:
            st.session_state.race_pending = True
            st.session_state.is_looping = True
            time.sleep(1.2)
            st.rerun()
    elif r_num:
        result_area.error("MKL run did not complete. Install `mkl_random`.")
        result_area.metric(f"NumPy ({numpy_bg})", f"{r_num.elapsed_s:.2f} s")
