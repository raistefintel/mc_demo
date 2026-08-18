"""Monte Carlo kernels — two backends, identical math, two workloads.

Workloads:
  * GBM path simulation — full price paths for a fan chart / VaR analysis.
  * European option pricing — single-batch Gaussian draw, MC price vs Black-Scholes
    closed form (mirrors the oneMKL sample:
    https://github.com/oneapi-src/oneMKL-samples/tree/main/monte_carlo_european_opt).

The ONLY difference between the two backends is the random number generator:
  * *_default -> numpy.random.Generator with a selectable BitGenerator
  * *_mkl     -> mkl_random.RandomState with a selectable BRNG

Defaults for both sides are MT19937 — the one algorithm implemented by both
libraries — so the head-to-head is a fair comparison of implementations. Users
can still switch either side independently to explore other engines.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

try:
    import mkl_random  # type: ignore
    HAS_MKL_RANDOM = True
except ImportError:
    HAS_MKL_RANDOM = False


# NumPy bit generators exposed through numpy.random.
NUMPY_BIT_GENERATORS: dict[str, type] = {
    "MT19937": np.random.MT19937,
    "PCG64": np.random.PCG64,
    "PCG64DXSM": np.random.PCG64DXSM,
    "SFC64": np.random.SFC64,
    "Philox": np.random.Philox,
}
DEFAULT_NUMPY_BG = "MT19937"

# Intel oneMKL VSL Basic Random Number Generators suitable for Gaussian sampling.
MKL_BRNGS: list[str] = [
    "MT19937",
    "SFMT19937",
    "MT2203",
    "MRG32K3A",
    "PHILOX4X32X10",
    "MCG59",
    "MCG31",
    "R250",
    "WH",
]
DEFAULT_MKL_BRNG = "MT19937"


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


# ---------------------------------------------------------------------------
# European option Monte Carlo (matches oneMKL sample structure).
# ---------------------------------------------------------------------------


def gaussian_stats(Z: np.ndarray) -> dict[str, float]:
    """Sample statistics of a Gaussian draw (used for cross-RNG equivalence checks)."""
    mean = float(Z.mean())
    std = float(Z.std())
    # Guard against std==0 (won't happen for large n but keeps this safe).
    skew = float(((Z - mean) ** 3).mean() / (std ** 3)) if std > 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": float(Z.min()),
        "max": float(Z.max()),
        "skew": skew,
    }


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    S0: float, K: float, r: float, sigma: float, T: float
) -> tuple[float, float]:
    """Analytical Black-Scholes prices for a European call and put."""
    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc = math.exp(-r * T)
    call = S0 * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    put = K * disc * _norm_cdf(-d2) - S0 * _norm_cdf(-d1)
    return float(call), float(put)


@dataclass
class OptionResult:
    call_price: float
    put_price: float
    call_bs: float
    put_bs: float
    stats: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    backend: str = ""
    n_paths: int = 0

    @property
    def call_error(self) -> float:
        return abs(self.call_price - self.call_bs)

    @property
    def put_error(self) -> float:
        return abs(self.put_price - self.put_bs)

    @property
    def l1_error(self) -> float:
        return self.call_error + self.put_error

    @property
    def options_per_s(self) -> float:
        return self.n_paths / self.elapsed_s if self.elapsed_s > 0 else 0.0


def _european_kernel(
    rng_normal: Callable[[int], np.ndarray],
    backend: str,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    progress_cb: Optional[ProgressCallback] = None,
) -> OptionResult:
    t0 = time.perf_counter()
    Z = rng_normal(n_paths)
    S_T = S0 * np.exp((r - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * Z)
    disc = math.exp(-r * T)
    call = float(disc * np.maximum(S_T - K, 0.0).mean())
    put = float(disc * np.maximum(K - S_T, 0.0).mean())
    elapsed = time.perf_counter() - t0

    call_bs, put_bs = black_scholes_price(S0, K, r, sigma, T)
    if progress_cb is not None:
        progress_cb(1, 1, elapsed)
    return OptionResult(
        call_price=call, put_price=put,
        call_bs=call_bs, put_bs=put_bs,
        stats=gaussian_stats(Z),
        elapsed_s=elapsed,
        backend=backend,
        n_paths=n_paths,
    )


def simulate_european_default(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int = 42,
    bit_generator: str = DEFAULT_NUMPY_BG,
    progress_cb: Optional[ProgressCallback] = None,
) -> OptionResult:
    """European option MC using a NumPy Generator with a selectable BitGenerator."""
    if bit_generator not in NUMPY_BIT_GENERATORS:
        raise ValueError(
            f"Unknown NumPy BitGenerator {bit_generator!r}. "
            f"Choose one of: {list(NUMPY_BIT_GENERATORS)}"
        )
    bg_cls = NUMPY_BIT_GENERATORS[bit_generator]
    rng = np.random.Generator(bg_cls(seed))
    return _european_kernel(
        rng_normal=lambda n: rng.standard_normal(n),
        backend=f"NumPy Generator ({bit_generator})",
        S0=S0, K=K, r=r, sigma=sigma, T=T, n_paths=n_paths,
        progress_cb=progress_cb,
    )


def simulate_european_mkl(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int = 42,
    brng: str = DEFAULT_MKL_BRNG,
    progress_cb: Optional[ProgressCallback] = None,
) -> OptionResult:
    """European option MC using Intel oneMKL VSL RNG via mkl_random."""
    if not HAS_MKL_RANDOM:
        raise RuntimeError(
            "mkl_random is not installed. Install with: pip install mkl-random"
        )
    rng = mkl_random.RandomState(seed, brng=brng)
    return _european_kernel(
        rng_normal=lambda n: rng.standard_normal(n),
        backend=f"Intel MKL VSL ({brng})",
        S0=S0, K=K, r=r, sigma=sigma, T=T, n_paths=n_paths,
        progress_cb=progress_cb,
    )
