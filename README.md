# Monte Carlo Stock Simulation — NumPy vs Intel oneMKL

A live side-by-side race showing the speedup of Intel's **oneMKL VSL** random
number generator over NumPy's default **Mersenne Twister**, using an identical
Geometric Brownian Motion (GBM) Monte Carlo kernel.

Same code. Same CPU. Same math. Only the RNG differs.

## What it shows

- **Two threads race in parallel** on the same Xeon:
  - Left: `numpy.random.default_rng()` — Mersenne Twister (MT19937)
  - Right: `mkl_random.RandomState(brng="SFMT19937")` — Intel oneMKL VSL
- Live throughput counters (M samples/sec) update every ~50 ms.
- Final speedup badge + financial output: mean final price, P(loss), 95% / 99% Value-at-Risk.
- Fan-of-paths chart and final-price histogram.

Typical speedup on Xeon: **8–15× faster** for the RNG step alone. If NumPy is
also Intel-distributed (MKL-linked), `np.exp` gains extra VML acceleration on top.

## Quick start

```bash
./run.sh
```

The script creates a local `.venv`, installs `numpy`, `mkl-random`, `streamlit`,
and `matplotlib`, then launches Streamlit at http://localhost:8501.

### Manual install
```bash
pip install -r requirements.txt
streamlit run app.py
```

## CLI benchmark (no UI)

```bash
python benchmark.py --paths 2000000 --steps 252 --repeats 5
```

Sample output:
```
GBM Monte Carlo — 2,000,000 paths × 252 steps = 504.0 M samples, 5 repeats

Backend                          Median (s)     Best (s)    M samples/s
------------------------------------------------------------------------
NumPy default (MT19937)              4.821        4.780           105.4
Intel oneMKL VSL (SFMT19937)         0.412        0.398          1266.3
------------------------------------------------------------------------

Speedup (median):  11.7×  (MKL over default)
```

## System info

```bash
python sysinfo.py
```
Reports NumPy build config, `mkl_random` version, MKL threads, and CPU model —
useful to confirm which BLAS/RNG is actually loaded.

## Files

- `mc_kernel.py` — GBM Monte Carlo, two backend functions (identical math).
- `app.py` — Streamlit live-race UI.
- `benchmark.py` — CLI benchmark harness.
- `sysinfo.py` — Environment introspection.
- `requirements.txt` — Python dependencies.
- `run.sh` — One-shot launcher script.

## Notes for demo presenters

- **Pin threads for a fair comparison.** By default both RNGs may use multiple
  threads. To compare single-threaded RNG-to-RNG, set
  `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before launching.
- **Use 1M+ paths.** With < 100k paths the elapsed time is too short to see the
  live progress bars move meaningfully.
- **`mkl_random` is independent of NumPy's BLAS.** It works with any NumPy
  (OpenBLAS or MKL-linked). The extra VML speedup on `np.exp` only kicks in if
  NumPy itself is MKL-linked (Intel Distribution for Python or the `intel`
  conda channel).
