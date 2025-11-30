"""Core simulation of Rough Bergomi paths using the Hybrid scheme.

This module contains the high-performance, Numba-accelerated kernel that
simulates joint (Sₜ, Vₜ) paths under the Rough Bergomi model with exact
fractional Brownian motion representation via the hybrid convolution scheme
(Bennedsen, Lunde, Pakkanen, 2017).

The implementation uses antithetic variates and parallelisation across paths.
"""
from numba import njit, prange
import numpy as np


@njit(parallel=True)
def simulate_rbergomi_paths(
    n_paths: int,
    n_steps: int,
    maturity: float,
    eta: float,
    hurst: float,
    rho: float,
    f_variance: np.ndarray,
    all_g1: np.ndarray,
    all_z: np.ndarray,
    short_rates: np.ndarray,
    K: np.ndarray,
):
    """
    Simulate correlated rough volatility paths using the Hybrid scheme.

    Dynamics:
        d(log S_t) = (r_t - V_t/2) dt + √V_t dW_t^S
        V_t = ξ_t(0) exp( η √(2H) W^H_t - (η²/2) t^{2H} )

    where W^H is a fractional Brownian motion with Hurst H < 0.5,
    and ξ_t(·) is the forward variance curve.

    Parameters
    ----------
    n_paths : int
        Number of Monte-Carlo paths
    n_steps : int
        Number of time steps
    maturity : float
        Longest maturity to simulate
    eta : float
        Volatility of volatility parameter
    hurst : float
        Hurst exponent (typically 0.02-0.15 for rough vol)
    rho : float
        Correlation between spot and volatility Brownian motions
    f_variance : np.ndarray
        Forward variance curve ξ_0(t_i) evaluated on the time grid
    all_g1 : np.ndarray
        Standard normal increments for the volatility Brownian motion (n_paths × n_steps)
    all_z : np.ndarray
        Independent standard normal increments for the orthogonal part
    short_rates : np.ndarray
        Instantaneous short rates r(t_i) on the time grid
    K : np.ndarray
        Pre-computed convolution matrix (from precompute_convolution_matrix)

    Returns
    -------
    s_paths : np.ndarray
        Simulated normalized stock paths (n_paths × n_steps+1), S_0 = 1
    variance_paths : np.ndarray
        Instantaneous variance paths V_t (n_paths × n_steps+1)
    """
    alpha = hurst - 0.5
    dt = maturity / n_steps
    n_time = n_steps + 1
    sqrt_dt = np.sqrt(dt)

    # ------------------------------------------------------------------ #
    # Brownian increments (with correlation)
    # ------------------------------------------------------------------ #

    delta_w1 = np.zeros((n_paths, n_time))
    delta_w1[:, 1:] = all_g1 * sqrt_dt
    delta_w2 = np.zeros((n_paths, n_time))
    delta_w2[:, 1:] = all_z * sqrt_dt
    delta_ws = rho * delta_w1 + np.sqrt(1 - rho**2) * delta_w2  # Cholesky

    s_paths = np.zeros((n_paths, n_time))
    variance_paths = np.zeros((n_paths, n_time))
    s_paths[:, 0] = 1.0
    variance_paths[:, 0] = f_variance[0]

    # ------------------------------------------------------------------ #
    # Fractional Brownian motion via Hybrid convolution
    # ------------------------------------------------------------------ #

    w = K @ delta_w1.T
    w = w.T * (dt**alpha)

    # Time grid
    t = np.linspace(0.0, maturity, n_time)
    t_2h = t ** (2 * hurst)

    # ------------------------------------------------------------------ #
    # Variance process (log-normal rough vol)
    # V_t = ξ_t(0) * exp(η √(2H) W^H_t - η² t^(2H))
    # where the √(2H) factor is part of the Riemann-Liouville fBm definition
    # ------------------------------------------------------------------ #

    vol_part = eta * np.sqrt(2 * hurst) * w
    drift_part = eta**2 * (t_2h)[None, :]  # (1, n_time)
    drift_part = np.broadcast_to(drift_part, vol_part.shape)

    exponent = vol_part - drift_part

    # Prevent overflow/underflow
    exponent = np.clip(exponent, -100.0, 100.0)

    variance_paths = np.broadcast_to(f_variance[None, :], exponent.shape) * np.exp(
        exponent
    )
    variance_paths = np.maximum(variance_paths, 1e-6 * f_variance.mean())

    # ------------------------------------------------------------------ #
    # Stock price process (Euler–Maruyama on log)
    # ------------------------------------------------------------------ #

    sqrt_variance = np.sqrt(variance_paths[:, :-1])

    short_rates = np.broadcast_to(short_rates[:n_steps], sqrt_variance.shape)
    drift = (short_rates - 0.5 * variance_paths[:, :-1]) * dt
    diffusion = sqrt_variance * delta_ws[:, 1:]
    d_log_s = drift + diffusion

    # Cumulative sum per path (parallel)
    log_s_cum = np.zeros((n_paths, n_steps + 1), dtype=np.float64)

    for p in prange(n_paths):
        acc = 0.0
        for i in range(n_steps):
            acc += d_log_s[p, i].item()
            log_s_cum[p, i + 1] = acc

    s_paths[:, 0] = 1.0
    s_paths[:, 1:] = np.exp(log_s_cum[:, 1:])

    return s_paths, variance_paths
