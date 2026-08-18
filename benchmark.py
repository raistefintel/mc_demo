"""Head-to-head CLI benchmark: NumPy default RNG vs Intel oneMKL VSL RNG.

Usage:
    python benchmark.py                          # defaults
    python benchmark.py --paths 2000000 --steps 252 --repeats 5
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
    simulate_gbm_default,
    simulate_gbm_mkl,
)


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
    p.add_argument("--numpy-bg", default=DEFAULT_NUMPY_BG,
                   choices=list(NUMPY_BIT_GENERATORS),
                   help="NumPy BitGenerator algorithm.")
    p.add_argument("--mkl-brng", default=DEFAULT_MKL_BRNG,
                   choices=MKL_BRNGS,
                   help="Intel oneMKL VSL BRNG algorithm.")
    args = p.parse_args()

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
            r = fn(**common, **extra)
            times.append(r.elapsed_s)
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
        print("\n[!] mkl_random not installed — skipping MKL run.")
        print("    Install with:  pip install mkl-random\n")


if __name__ == "__main__":
    main()
