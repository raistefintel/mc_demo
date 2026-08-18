"""Geometric Brownian Motion Monte Carlo — two backends, identical math.

Both simulators run the same GBM update:
    S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt) * Z),  Z ~ N(0,1)

The ONLY difference between the two functions is the random number generator:
  * simulate_gbm_default -> numpy.random.Generator with a selectable BitGenerator
  * simulate_gbm_mkl     -> mkl_random.RandomState with a selectable BRNG

Everything else (np.exp, arithmetic) uses whatever NumPy is installed. If NumPy
itself is Intel-distributed / MKL-linked, np.exp will use MKL VML on top, which
adds even more speedup — but the RNG swap alone is what this demo highlights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import mkl_random  # type: ignore
    HAS_MKL_RANDOM = True
except ImportError:
    HAS_MKL_RANDOM = False


# NumPy bit generators exposed through numpy.random.
NUMPY_BIT_GENERATORS: dict[str, type] = {
    "PCG64": np.random.PCG64,
    "PCG64DXSM": np.random.PCG64DXSM,
    "MT19937": np.random.MT19937,
    "SFC64": np.random.SFC64,
    "Philox": np.random.Philox,
}
DEFAULT_NUMPY_BG = "PCG64"

# Intel oneMKL VSL Basic Random Number Generators suitable for Gaussian sampling.
MKL_BRNGS: list[str] = [
    "SFMT19937",
    "MT19937",
    "MT2203",
    "MRG32K3A",
    "PHILOX4X32X10",
    "MCG59",
    "MCG31",
    "R250",
    "WH",
]
DEFAULT_MKL_BRNG = "SFMT19937"


@dataclass
class SimResult:
    paths: np.ndarray          # shape (n_steps + 1, n_paths)
    elapsed_s: float
    backend: str
    n_paths: int
    n_steps: int

    @property
    def throughput_samples_per_s(self) -> float:
        # Total random samples drawn = n_steps * n_paths
        return (self.n_steps * self.n_paths) / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def throughput_paths_per_s(self) -> float:
        return self.n_paths / self.elapsed_s if self.elapsed_s > 0 else 0.0


ProgressCallback = Callable[[int, int, float], None]
# args: (current_step, total_steps, elapsed_seconds)


def _gbm_loop(
    rng_normal: Callable[[int], np.ndarray],
    backend: str,
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    progress_cb: Optional[ProgressCallback] = None,
    progress_every: int = 4,
) -> SimResult:
    dt = T / n_steps
    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)

    paths = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    paths[0] = S0
    log_S = np.full(n_paths, np.log(S0), dtype=np.float64)

    t0 = time.perf_counter()
    for t in range(1, n_steps + 1):
        Z = rng_normal(n_paths)
        log_S += drift + diffusion * Z
        # np.exp here benefits from MKL VML if NumPy is MKL-linked.
        paths[t] = np.exp(log_S)
        if progress_cb is not None and (t % progress_every == 0 or t == n_steps):
            progress_cb(t, n_steps, time.perf_counter() - t0)
    elapsed = time.perf_counter() - t0

    return SimResult(
        paths=paths,
        elapsed_s=elapsed,
        backend=backend,
        n_paths=n_paths,
        n_steps=n_steps,
    )


def simulate_gbm_default(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
    bit_generator: str = DEFAULT_NUMPY_BG,
    progress_cb: Optional[ProgressCallback] = None,
) -> SimResult:
    """GBM Monte Carlo using a NumPy Generator with a selectable BitGenerator."""
    if bit_generator not in NUMPY_BIT_GENERATORS:
        raise ValueError(
            f"Unknown NumPy BitGenerator {bit_generator!r}. "
            f"Choose one of: {list(NUMPY_BIT_GENERATORS)}"
        )
    bg_cls = NUMPY_BIT_GENERATORS[bit_generator]
    rng = np.random.Generator(bg_cls(seed))
    return _gbm_loop(
        rng_normal=lambda n: rng.standard_normal(n),
        backend=f"NumPy Generator ({bit_generator})",
        S0=S0, mu=mu, sigma=sigma, T=T,
        n_steps=n_steps, n_paths=n_paths,
        progress_cb=progress_cb,
    )


def simulate_gbm_mkl(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
    brng: str = DEFAULT_MKL_BRNG,
    progress_cb: Optional[ProgressCallback] = None,
) -> SimResult:
    """GBM Monte Carlo using Intel oneMKL VSL RNG via mkl_random."""
    if not HAS_MKL_RANDOM:
        raise RuntimeError(
            "mkl_random is not installed. Install with: pip install mkl-random"
        )
    rng = mkl_random.RandomState(seed, brng=brng)
    return _gbm_loop(
        rng_normal=lambda n: rng.standard_normal(n),
        backend=f"Intel MKL VSL ({brng})",
        S0=S0, mu=mu, sigma=sigma, T=T,
        n_steps=n_steps, n_paths=n_paths,
        progress_cb=progress_cb,
    )


def value_at_risk(final_prices: np.ndarray, S0: float, confidence: float = 0.95) -> float:
    """Compute Value-at-Risk (loss at the given confidence level) in currency units."""
    losses = S0 - final_prices
    return float(np.quantile(losses, confidence))
