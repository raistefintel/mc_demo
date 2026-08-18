"""Streamlit app: NumPy built-in RNG vs Intel oneMKL RNG.

Two workloads:
  * European option pricing (default) — single-batch Gaussian draw with a
    Black-Scholes reference; mirrors the oneMKL sample.
  * GBM path simulation — full price paths + VaR.

Runs execute sequentially — NumPy on the left panel, mkl_random on the right —
so each backend gets exclusive CPU time. A live ticker updates elapsed seconds
every ~50 ms while a worker thread crunches numbers in the background.
"""

from __future__ import annotations

import os
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


def render_panel(container, label, prog, n_paths, threads, status):
    title, subtitle, color = label
    live = status == "running" and prog.start_time is not None and not prog.done
    elapsed = (time.perf_counter() - prog.start_time) if live else prog.elapsed
    samples = prog.step * n_paths
    tput = samples / prog.elapsed if prog.elapsed > 0 else 0.0

    container.markdown(f"### {title}")
    container.caption(subtitle)
    container.markdown(
        f"<h1 style='text-align:center;color:{'#888' if status == 'idle' else color};"
        f"margin:0;font-family:monospace;font-size:4rem'>"
        f"{'—.—— s' if status == 'idle' else f'{elapsed:6.2f} s'}</h1>",
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
        "European option pricing mirrors the Intel oneMKL sample: "
        "https://github.com/oneapi-src/oneMKL-samples/tree/main/monte_carlo_european_opt\n\n"
        "GBM path simulation draws n_steps × n_paths Gaussians and produces full paths."
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
st.sidebar.caption(
    f"mkl_random: {'✓' if HAS_MKL_RANDOM else '✗'} · "
    f"mkl-service: {'✓' if HAS_MKL_SERVICE else '✗'} · "
    f"CPU cores: {cpu_count} · NumPy: {np.__version__}"
)

# ---------------- Main ----------------
NUMPY_LABEL = (
    "1. NumPy built-in RNG",
    f"numpy.random.Generator(np.random.{numpy_bg}(seed))",
    "#0068C9",
)
MKL_LABEL = (
    "2. Intel oneMKL RNG",
    f"mkl_random.RandomState(seed, brng='{mkl_brng}')",
    "#FF8C00",
)

st.title(
    "European Option Pricing — NumPy vs Intel oneMKL"
    if IS_OPTION else
    "Monte Carlo Stock Simulation — NumPy vs Intel oneMKL"
)
if IS_OPTION:
    st.caption(
        "Monte Carlo European call & put pricing under Black-Scholes. "
        "MC results are validated against the closed-form Black-Scholes price. "
        "Mirrors the [Intel oneMKL sample]"
        "(https://github.com/oneapi-src/oneMKL-samples/tree/main/monte_carlo_european_opt)."
    )
else:
    st.caption(
        "Identical Geometric Brownian Motion kernel. The two runs execute "
        "**sequentially** — first with NumPy's built-in RNG (left), then with "
        "Intel's `mkl_random` (right). The live ticker shows elapsed seconds."
    )

start = st.button("▶ Run comparison", type="primary", use_container_width=True)

col_left, col_right = st.columns(2)
panel_left, panel_right = col_left.empty(), col_right.empty()
result_area = st.container()

if not HAS_MKL_RANDOM:
    st.warning("`mkl_random` not installed — MKL side will error. `pip install mkl-random`")

# Initial idle render (persists in the placeholder until overwritten).
idle = Progress()
with panel_left.container():
    render_panel(st, NUMPY_LABEL, idle, int(n_paths), 1, "idle")
with panel_right.container():
    render_panel(st, MKL_LABEL, idle, int(n_paths), mkl_threads, "idle")

if start and IS_OPTION:
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
        result_area.markdown("## Results")

        c1, c2, c3 = result_area.columns(3)
        c1.metric(f"NumPy ({numpy_bg})", f"{r_num.elapsed_s:.3f} s",
                  f"{r_num.options_per_s/1e6:.1f} M options/s")
        c2.metric(f"Intel oneMKL ({mkl_brng})", f"{r_mkl.elapsed_s:.3f} s",
                  f"{r_mkl.options_per_s/1e6:.1f} M options/s")
        c3.metric("Speedup", f"{speedup:.1f}×", "MKL vs NumPy")

        result_area.markdown("### Correctness vs Black-Scholes closed form")
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

        result_area.markdown("### Sanity check: Gaussian sample statistics")
        result_area.caption(
            "Both RNGs should produce N(0,1) samples with matching statistics — "
            "confirms mathematical equivalence up to sample noise."
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

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        bar_labels = ["Black-Scholes", f"NumPy\n({numpy_bg})", f"MKL\n({mkl_brng})"]
        bar_colors = ["black", "#0068C9", "#FF8C00"]
        ax[0].bar(bar_labels, [call_bs, r_num.call_price, r_mkl.call_price], color=bar_colors)
        ax[0].set(title="European CALL price", ylabel="Price ($)")
        ax[0].grid(axis="y", alpha=0.3)
        ax[1].bar(bar_labels, [put_bs, r_num.put_price, r_mkl.put_price], color=bar_colors)
        ax[1].set(title="European PUT price", ylabel="Price ($)")
        ax[1].grid(axis="y", alpha=0.3)
        fig.tight_layout()
        result_area.pyplot(fig)
    elif r_num:
        result_area.error("MKL run did not complete. Install `mkl_random`.")
        result_area.metric(f"NumPy ({numpy_bg})", f"{r_num.elapsed_s:.3f} s")

elif start:
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
        result_area.markdown("## Result")
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
    elif r_num:
        result_area.error("MKL run did not complete. Install `mkl_random`.")
        result_area.metric(f"NumPy ({numpy_bg})", f"{r_num.elapsed_s:.2f} s")
