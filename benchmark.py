"""Head-to-head CLI benchmark: NumPy default RNG vs Intel oneMKL VSL RNG.

Usage:
    python benchmark.py                          # defaults
    python benchmark.py --paths 2000000 --steps 252 --repeats 5
"""

from __future__ import annotations

import argparse
import statistics

from mc_kernel import HAS_MKL_RANDOM, simulate_gbm_default, simulate_gbm_mkl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", type=int, default=1_000_000)
    p.add_argument("--steps", type=int, default=252)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--mu", type=float, default=0.08)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    common = dict(
        S0=args.S0, mu=args.mu, sigma=args.sigma, T=args.T,
        n_steps=args.steps, n_paths=args.paths, seed=args.seed,
    )
    total_samples = args.steps * args.paths

    print(f"\nGBM Monte Carlo — {args.paths:,} paths × {args.steps} steps "
          f"= {total_samples/1e6:.1f} M samples, {args.repeats} repeats\n")
    print(f"{'Backend':<32} {'Median (s)':>12} {'Best (s)':>12} {'M samples/s':>14}")
    print("-" * 72)

    def run(name, fn):
        times = []
        # Warmup
        fn(**common)
        for _ in range(args.repeats):
            r = fn(**common)
            times.append(r.elapsed_s)
        med = statistics.median(times)
        best = min(times)
        tput = total_samples / best / 1e6
        print(f"{name:<32} {med:>12.3f} {best:>12.3f} {tput:>14.1f}")
        return med

    med_def = run("NumPy default (MT19937)", simulate_gbm_default)

    if HAS_MKL_RANDOM:
        med_mkl = run("Intel oneMKL VSL (SFMT19937)", simulate_gbm_mkl)
        print("-" * 72)
        print(f"\nSpeedup (median):  {med_def / med_mkl:.2f}×  (MKL over default)\n")
    else:
        print("\n[!] mkl_random not installed — skipping MKL run.")
        print("    Install with:  pip install mkl-random\n")


if __name__ == "__main__":
    main()
