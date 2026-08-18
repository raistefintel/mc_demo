# Monte Carlo Financial Benchmark — NumPy vs Intel oneMKL

A live side-by-side race showing the speedup of Intel's **oneMKL VSL** random
number generator over NumPy's built-in RNG on **two industry-standard Monte
Carlo workloads**:

1. **European option pricing** (default) — matches the [Intel oneMKL sample][mkl-sample]
   with a Black-Scholes closed-form reference for correctness validation.
2. **GBM path simulation** — full price paths for Value-at-Risk / fan-chart analysis.

Same code. Same CPU. Same math. **Only the RNG differs**, and by default both
sides use **identical MT19937** so the comparison is strictly apples-to-apples.

[mkl-sample]: https://github.com/oneapi-src/oneMKL-samples/tree/main/monte_carlo_european_opt

## What it shows

- **Sequential head-to-head**: NumPy on the left panel, `mkl_random` on the right.
- **Live throughput counter** (M options/sec or M samples/sec) with an elapsed-time ticker.
- **Correctness validation** (option workload): MC prices are compared to the
  Black-Scholes closed form — TEST PASSED banner when both sides converge within
  tolerance.
- **Gaussian sanity table**: mean / std / min / max / skew of both RNGs' samples
  side-by-side to confirm mathematical equivalence.
- **Speedup badge** and matching-engine caption when both sides use MT19937.

Typical speedup on Xeon: **5–15× faster** for the option pricing workload,
depending on MKL thread count. Using **PCG64 / SFC64 / SFMT19937 / PHILOX4X32X10**
via the dropdowns exposes further per-engine performance differences.

## Quick start

```bash
git clone https://github.com/raistefintel/mc_demo.git
cd mc_demo
chmod +x run.sh
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
# European option pricing (default) — reports options/sec + L1 error vs Black-Scholes
python benchmark.py

# GBM path simulation — reports samples/sec
python benchmark.py --workload path --paths 2000000 --steps 252 --repeats 5

# Choose engines explicitly
python benchmark.py --numpy-bg PCG64 --mkl-brng SFMT19937
```

Sample output (European workload):
```
European option MC — 10,000,000 paths, 3 repeats
  S0=100.0  K=100.0  r=0.05  sigma=0.2  T=1.0
  Black-Scholes:  call = $10.4506   put = $5.5735

Backend                              Median (s)     Best (s)    M opt/s     L1 err
----------------------------------------------------------------------------------
NumPy Generator (MT19937)                 0.412        0.398       25.1     0.0037
Intel oneMKL VSL (MT19937)                0.061        0.058      172.4     0.0031
----------------------------------------------------------------------------------

Speedup (median):  6.75×  (MKL over NumPy)
```

## System info

```bash
python sysinfo.py
```
Reports NumPy build config, `mkl_random` version, MKL threads, and CPU model —
useful to confirm which BLAS/RNG is actually loaded.

## Files

- `mc_kernel.py` — GBM path + European option Monte Carlo kernels; Black-Scholes
  reference; Gaussian sample-statistics helper.
- `app.py` — Streamlit UI with workload selector, sequential head-to-head,
  correctness panel, and Gaussian sanity table.
- `benchmark.py` — CLI benchmark harness (`--workload european | path`).
- `sysinfo.py` — Environment introspection.
- `requirements.txt` — Python dependencies.
- `run.sh` — One-shot launcher script.

## Fair-comparison defaults

Both dropdowns default to **`MT19937`** — the exact same Mersenne Twister
algorithm implemented by both `numpy.random` and `mkl_random`. This isolates
what is being measured to the **vendor implementation quality**, not
algorithm differences.

The UI still lets you pick any engine on either side:

| Engine | NumPy | mkl_random |
|---|---|---|
| MT19937 (default) | ✓ | ✓ |
| PCG64 / PCG64DXSM / SFC64 / Philox | ✓ | — |
| SFMT19937 / MT2203 / MRG32K3A / PHILOX4X32X10 / MCG59 / MCG31 / R250 / WH | — | ✓ |

## Notes for demo presenters

- **Correctness proof matters as much as speed.** In European mode the demo shows
  MC prices matching Black-Scholes to ~4 decimals, with a `TEST PASSED` banner.
  This is what the reviewer at Intel oneMKL flagged as the industry-standard
  quality gate.
- **Use ≥ 10 M paths for the option workload.** The MC standard error scales as
  `1/√n`, so 10 M paths gets ~4 significant digits. Also gives the multi-threaded
  MKL RNG enough work to shine on a Xeon.
- **`mkl_random` is independent of NumPy's BLAS.** It works with any NumPy
  (OpenBLAS or MKL-linked). Extra VML speedup on `np.exp` only kicks in if
  NumPy itself is MKL-linked (Intel Distribution for Python or the `intel`
  conda channel).
