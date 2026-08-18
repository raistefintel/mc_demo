"""Head-to-head CLI benchmark: NumPy default RNG vs Intel oneMKL VSL RNG.

Usage:
    python benchmark.py                                # European option (default)
    python benchmark.py --workload path                # GBM path simulation
    python benchmark.py --paths 10000000 --repeats 5   # override sizes
"""

from __future__ import annotations

import argparse
import statistics

from mc_kernel import (
    DEFAULT_MKL_BRNG,
    DEFAULT_NUMPY_BG,
    HAS_MKL_RANDOM,
    MKL_BRNGS,
    NUMPY_BIT_GENERATORS,
    black_scholes_price,
    simulate_european_default,
    simulate_european_mkl,
    simulate_gbm_default,
    simulate_gbm_mkl,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", default="european",
                   choices=["european", "path"],
                   help="european: option pricing (default). path: GBM full-path sim.")
    p.add_argument("--paths", type=int, default=10_000_000)
    p.add_argument("--steps", type=int, default=252, help="Only used for --workload path.")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--K", type=float, default=100.0, help="Strike (european workload).")
    p.add_argument("--r", type=float, default=0.05, help="Risk-free rate (european workload).")
    p.add_argument("--mu", type=float, default=0.08, help="Drift (path workload).")
    p.add_argument("--sigma", type=float, default=0.20)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--numpy-bg", default=DEFAULT_NUMPY_BG,
                   choices=list(NUMPY_BIT_GENERATORS),
                   help="NumPy BitGenerator algorithm.")
    p.add_argument("--mkl-brng", default=DEFAULT_MKL_BRNG,
                   choices=MKL_BRNGS,
                   help="Intel oneMKL VSL BRNG algorithm.")
    args = p.parse_args()

    if args.workload == "european":
        _run_european(args)
    else:
        _run_path(args)


def _run_european(args) -> None:
    common = dict(
        S0=args.S0, K=args.K, r=args.r, sigma=args.sigma, T=args.T,
        n_paths=args.paths, seed=args.seed,
    )
    call_bs, put_bs = black_scholes_price(args.S0, args.K, args.r, args.sigma, args.T)

    print(f"\nEuropean option MC — {args.paths:,} paths, {args.repeats} repeats")
    print(f"  S0={args.S0}  K={args.K}  r={args.r}  sigma={args.sigma}  T={args.T}")
    print(f"  Black-Scholes:  call = ${call_bs:.4f}   put = ${put_bs:.4f}\n")
    print(f"{'Backend':<36} {'Median (s)':>12} {'Best (s)':>12} {'M opt/s':>10} {'L1 err':>10}")
    print("-" * 82)

    def run(name, fn, extra):
        times, l1s = [], []
        fn(**common, **extra)  # warmup
        for _ in range(args.repeats):
            r_ = fn(**common, **extra)
            times.append(r_.elapsed_s)
            l1s.append(r_.l1_error)
        med = statistics.median(times)
        best = min(times)
        tput = args.paths / best / 1e6
        l1_med = statistics.median(l1s)
        print(f"{name:<36} {med:>12.3f} {best:>12.3f} {tput:>10.1f} {l1_med:>10.4f}")
        return med

    med_def = run(f"NumPy Generator ({args.numpy_bg})",
                  simulate_european_default, {"bit_generator": args.numpy_bg})
    if HAS_MKL_RANDOM:
        med_mkl = run(f"Intel oneMKL VSL ({args.mkl_brng})",
                      simulate_european_mkl, {"brng": args.mkl_brng})
        print("-" * 82)
        print(f"\nSpeedup (median):  {med_def / med_mkl:.2f}×  (MKL over NumPy)\n")
    else:
        print("\n[!] mkl_random not installed — skipping MKL run.\n")


def _run_path(args) -> None:
    common = dict(
        S0=args.S0, mu=args.mu, sigma=args.sigma, T=args.T,
        n_steps=args.steps, n_paths=args.paths, seed=args.seed,
    )
    total_samples = args.steps * args.paths

    print(f"\nGBM Monte Carlo — {args.paths:,} paths × {args.steps} steps "
          f"= {total_samples/1e6:.1f} M samples, {args.repeats} repeats\n")
    print(f"{'Backend':<36} {'Median (s)':>12} {'Best (s)':>12} {'M samples/s':>14}")
    print("-" * 76)

    def run(name, fn, extra):
        times = []
        fn(**common, **extra)  # warmup
        for _ in range(args.repeats):
            r_ = fn(**common, **extra)
            times.append(r_.elapsed_s)
        med = statistics.median(times)
        best = min(times)
        tput = total_samples / best / 1e6
        print(f"{name:<36} {med:>12.3f} {best:>12.3f} {tput:>14.1f}")
        return med

    med_def = run(f"NumPy Generator ({args.numpy_bg})",
                  simulate_gbm_default, {"bit_generator": args.numpy_bg})
    if HAS_MKL_RANDOM:
        med_mkl = run(f"Intel oneMKL VSL ({args.mkl_brng})",
                      simulate_gbm_mkl, {"brng": args.mkl_brng})
        print("-" * 76)
        print(f"\nSpeedup (median):  {med_def / med_mkl:.2f}×  (MKL over NumPy)\n")
    else:
        print("\n[!] mkl_random not installed — skipping MKL run.\n")


if __name__ == "__main__":
    main()
