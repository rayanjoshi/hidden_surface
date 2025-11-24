"""
Rough Bergomi (rBergomi) Model Implementation
============================================

A high-performance, Numba-accelerated implementation of the Rough Bergomi stochastic
volatility model for:

- Simulation of forward variance and stock price paths under rough volatility dynamics
- Monte-Carlo pricing of European vanilla options
- Global calibration to implied volatility surfaces (calls + puts)
- Hybrid scheme convolution for exact fractional Brownian motion representation

Key Features
------------
- Uses the Hybrid convolution scheme (Bennedsen et al.) for accurate simulation of
  fractional Brownian motion with arbitrary Hurst parameter H ∈ (0, 0.5)
- Antithetic variates for variance reduction
- Parallel Numba kernels for both variance and stock processes
- Robust calibration using trust-region reflective least-squares with regularization
- Full support for time-dependent short rates and forward variance curves

References
----------
- Rough Bergomi (2008) - Christian Bayer, Peter Friz, Jim Gatheral
- "Hybrid scheme for Brownian semistationary processes" - Bennedsen, Lunde, Pakkanen (2017)

Dependencies:
    - numpy
    - pandas
    - scipy
    - numba
    - matplotlib
    - hydra
    - omegaconf
"""

from typing import Optional, Dict
from pathlib import Path
import json
import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import interp1d, splev, splrep
from scipy.optimize import least_squares, OptimizeResult
import hydra
from omegaconf import DictConfig
from numba import njit, prange
import matplotlib.pyplot as plt

from scripts.logging_config import get_logger, setup_logging

# =============================================================================
# Black-Scholes Helper Functions
# =============================================================================


def black_scholes_call(
    s: np.ndarray | float,
    k: np.ndarray | float,
    t: np.ndarray | float,
    r: float = 0,
    q: float = 0,
    sigma: float | np.ndarray = 0.2,
) -> np.ndarray:
    """
    Vectorized Black-Scholes call option price.

    Handles edge cases (zero volatility, zero time) gracefully to avoid NaNs
    during implied volatility calculations.

    Parameters
    ----------
    s : array_like
        Spot price(s)
    k : array_like
        Strike price(s)
    t : array_like
        Time to maturity (years)
    r : float
        Risk-free rate
    q : float
        Continuous dividend yield
    sigma : float or array_like
        Volatility

    Returns
    -------
        np.ndarray
        Call prices with the same shape as broadcasted inputs
    """
    s = np.asarray(s)
    k = np.asarray(k)
    t = np.asarray(t)
    sigma = np.where(sigma < 1e-10, 1e-10, sigma)  # avoid division by zero

    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t + 1e-10)
    d2 = d1 - sigma * sqrt_t
    call_price = s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

    # Deterministic limit when volatility collapses
    call_price = np.where(
        sigma < 1e-10, np.maximum(s - k, 0.0) * np.exp(-r * t), call_price
    )
    return call_price


def black_scholes_put(
    s: np.ndarray | float,
    k: np.ndarray | float,
    t: np.ndarray | float,
    r: float = 0,
    q: float = 0,
    sigma: float | np.ndarray = 0.2,
):
    """
    Vectorized Black-Scholes put option price.

    Parameters
    ----------
    s : array_like
        Spot price(s)
    k : array_like
        Strike price(s)
    t : array_like
        Time to maturity (years)
    r : float
        Risk-free rate
    q : float
        Continuous dividend yield
    sigma : float or array_like
        Volatility

    Returns
    -------
        np.ndarray
        Call prices with the same shape as broadcasted inputs
    """
    s = np.asarray(s)
    k = np.asarray(k)
    t = np.asarray(t)
    sigma = np.where(sigma < 1e-10, 1e-10, sigma)

    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t + 1e-10)
    d2 = d1 - sigma * sqrt_t
    put_price = k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1)

    put_price = np.where(
        sigma < 1e-10, np.maximum(k - s, 0.0) * np.exp(-r * t), put_price
    )
    return put_price


def implied_volatility(
    price: float,
    s: float,
    k: float,
    t: float,
    r: float = 0,
    q: float = 0,
    option_type: str = "call",
) -> float:
    """
    Compute Black-Scholes implied volatility using Brent's method.

    Returns NaN on failure (e.g. arbitrage violations).

    Parameters
    ----------
    price : float
        Observed market price
    s, k, t, r, q : float
        Standard Black-Scholes inputs
    option_type : {'call', 'put'}

    Returns
    -------
    float
        Implied volatility, or np.nan if solver fails
    """
    if price <= 0 or np.isnan(price):
        return np.nan

    def objective(sigma: float) -> float:
        if option_type == "call":
            model_price = black_scholes_call(s, k, t, r, q, sigma)
        else:
            model_price = black_scholes_put(s, k, t, r, q, sigma)
        return float(np.atleast_1d(model_price)[0] - price)

    try:
        res = brentq(objective, 1e-6, 20.0)
        if isinstance(res, tuple):
            res = res[0]
        return float(res)
    except ValueError:
        return np.nan


# =============================================================================
# Rough Bergomi Simulation Core
# =============================================================================


@njit
def precompute_convolution_matrix(alpha: float, n_steps: int) -> np.ndarray:
    """
    Pre-compute the lower-triangular convolution kernel matrix K for the Hybrid scheme.

    The kernel is given by:

        K_{i,j} = (i - j + 1)^{α+1} - (i - j)^{α+1}   for j ≤ i
                  ---------------------------------
                                     α + 1

    where α = H - 1/2 and H is the Hurst parameter.

    Parameters
    ----------
    alpha : float
        α = H - 0.5  (must be in (-0.5, 0) for rough vol)
    n_steps : int
        Number of time steps (matrix size = n_steps + 1)

    Returns
    -------
    np.ndarray
        (n_steps+1, n_steps+1) lower-triangular convolution matrix
    """
    if abs(alpha + 1) < 1e-10:
        raise ValueError("alpha + 1 too close to zero")
    n = n_steps + 1
    K = np.zeros((n, n))
    kernel = np.zeros(n)

    # Compute fractional kernel coefficients
    for m in range(1, n):
        kernel[m] = (m + 1) ** (alpha + 1) - m ** (alpha + 1)
    kernel[1:] /= alpha + 1

    # Fill lower-triangular part
    for i in range(n):
        for j in range(i + 1):
            K[i, j] = kernel[i - j]
    return K


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

    w = np.zeros((n_paths, n_time))
    for p in prange(n_paths):
        w[p] = K @ delta_w1[p]
    w = w * (dt**alpha)  # scale w by dt**alpha

    # # Time grid (Numba-compatible)
    t = np.empty(n_time)
    for i in range(n_time):
        t[i] = i * dt
    t_2h = t ** (2 * hurst)

    # Tile time vector for vectorized operations
    t_2h_tile = np.zeros((n_paths, n_time))
    for p in prange(n_paths):
        for i in range(n_time):
            t_2h_tile[p, i] = t_2h[i]

    # ------------------------------------------------------------------ #
    # Variance process (log-normal rough vol)
    # V_t = ξ_t(0) * exp(η √(2H) W^H_t - (1/2) η² t^(2H))
    # where the √(2H) factor is part of the Riemann-Liouville fBm definition
    # ------------------------------------------------------------------ #

    exponent = eta * np.sqrt(2 * hurst) * w - 0.5 * (eta**2) * t_2h_tile
    # Prevent overflow/underflow
    for p in prange(n_paths):
        for i in range(n_time):
            if exponent[p, i] > 100.0:
                exponent[p, i] = 100.0
            elif exponent[p, i] < -100.0:
                exponent[p, i] = -100.0

    f_variance_tile = np.zeros((n_paths, n_time))
    for p in prange(n_paths):
        for i in range(n_time):
            f_variance_tile[p, i] = f_variance[i]

    # Floor variance to avoid numerical issues
    variance_paths = f_variance_tile * np.exp(exponent)
    min_variance = 1e-6 * np.mean(f_variance)
    for p in prange(n_paths):
        for i in range(n_time):
            if variance_paths[p, i] < min_variance:
                variance_paths[p, i] = min_variance

    # ------------------------------------------------------------------ #
    # Stock price process (Euler–Maruyama on log)
    # ------------------------------------------------------------------ #

    sqrt_variance = np.sqrt(variance_paths[:, :-1])
    short_rates_tile = np.zeros((n_paths, n_steps))
    for p in prange(n_paths):
        for i in range(n_steps):
            short_rates_tile[p, i] = short_rates[i]

    drift = (short_rates_tile - 0.5 * variance_paths[:, :-1]) * dt
    diffusion = sqrt_variance * delta_ws[:, 1:]
    d_log_s = drift + diffusion

    # Cumulative sum per path (parallel)
    log_s_cum = np.zeros_like(d_log_s)
    for p in prange(n_paths):
        cum = 0.0
        for i in range(n_steps):
            cum += d_log_s[p, i]
            log_s_cum[p, i] = cum

    for p in prange(n_paths):
        for i in range(1, n_time):
            s_paths[p, i] = np.exp(log_s_cum[p, i - 1])

    return s_paths, variance_paths


# =============================================================================
# Pricing & Calibration Engine
# =============================================================================


class RoughBergomiEngine:
    """
    Core engine for Rough Bergomi pricing and calibration.

    Handles:
        - Loading of forward variance curve ξ_t(T)
        - Loading of implied volatility surface
        - Interpolation of risk-free rates and short rates
        - Monte-Carlo pricing with antithetic variates
        - Global least-squares calibration with regularization
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the engine from Hydra configuration.

        Loads all required market data (forward variance, IV surface, rates)
        and builds interpolators.
        """
        self.cfg = cfg
        self.logger = get_logger("RoughBergomiEngine")
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        f_variance_path = Path(repo_root / self.cfg.f_variance.output_path).resolve()
        self.df = pd.read_csv(f_variance_path)

        # --------------------- Forward variance curve --------------------- #
        self.forward_variance_func = interp1d(
            self.df["maturity_years"],
            self.df["forward_variance"],
            bounds_error=False,
            fill_value=(
                float(self.df["forward_variance"].iloc[0]),
                float(self.df["forward_variance"].iloc[-1])
            ),
        )

        # --------------------- Implied volatility surface --------------------- #
        iv_surface_path = Path(repo_root / self.cfg.iv_surface.surface_path).resolve()
        self.iv_surface = np.load(iv_surface_path)
        if np.any(np.isnan(self.iv_surface)) or np.any(self.iv_surface <= 0):
            raise ValueError("IV surface contains NaN or non-positive values")

        maturities_path = Path(
            repo_root / self.cfg.iv_surface.maturities_path
        ).resolve()
        self.maturities = np.load(maturities_path)
        log_moneyness_path = Path(
            repo_root / self.cfg.iv_surface.log_moneyness_path
        ).resolve()
        self.log_moneyness = np.load(log_moneyness_path)
        if np.any(np.isnan(self.log_moneyness)):
            raise ValueError("Log moneyness contain NaN values")
        self.strikes = np.exp(self.log_moneyness)

        # --------------------- Risk-free rate curve --------------------- #
        rf_path = self.cfg.iv_surface.risk_free_rate_path
        self.risk_free_rate_path = Path(repo_root / rf_path).resolve()
        self.risk_free_rate = np.load(self.risk_free_rate_path)

        if np.any(np.isnan(self.maturities)) or np.any(self.maturities <= 0):
            raise ValueError("Maturities contain NaN or non-positive values")

        risk_free_rates = self.risk_free_rate.copy()
        # Ensure spline works at t=0
        if self.maturities[0] > 0:
            self.maturities = np.insert(self.maturities, 0, 0)
            risk_free_rates = np.insert(self.risk_free_rate, 0, self.risk_free_rate[0])
        self.yield_spl = splrep(self.maturities, risk_free_rates, s=0, k=3)

        if (
            self.df["maturity_years"].isnull().any()
            or self.df["forward_variance"].isnull().any()
        ):
            raise ValueError("Forward variance data contains NaN values")
        if (self.df["forward_variance"] <= 0).any():
            raise ValueError("Forward variance contains non-positive values")

        start = self.df["maturity_years"].min()
        end = self.df["maturity_years"].max()
        self.logger.info("Forward variance data range: %s to %s", start, end)

    # --------------------------------------------------------------------- #
    # Yield & short rate interpolation
    # --------------------------------------------------------------------- #

    def get_yield(self, t: float | np.ndarray) -> float | np.ndarray:
        """Zero-coupon yield Y(t) = -ln(P(0,t))/t"""
        res = splev(t, self.yield_spl, ext=0)
        res_arr = np.asarray(res)
        if np.ndim(res_arr) == 0:
            return float(res_arr)
        return res_arr

    def get_short_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Instantaneous short rate r(t) = Y(t) + t * dY/dt(t)
        Computed via spline derivative.
        """
        if np.isscalar(t):
            # Handle t=0 case directly
            if t == 0.0:
                rate = float(np.atleast_1d(splev(1e-10, self.yield_spl, ext=0))[0])
                return rate
            try:
                # Convert to a numpy scalar/array first and extract a Python scalar
                t_arr = np.asarray(t)
                # If complex-valued, use the real part to avoid float(complex) errors
                if np.iscomplexobj(t_arr):
                    t_val = float(np.real(t_arr).item())
                else:
                    t_val = float(t_arr.item())
            except (TypeError, ValueError, IndexError, AttributeError):
                t_val = 1e-6
            if t_val <= 0.0:
                t_val = 1e-6
            rate = splev(t_val, self.yield_spl, ext=0)
            d_rate_dt = splev(t_val, self.yield_spl, der=1, ext=0)
            rate_arr = np.atleast_1d(rate)
            d_rate_dt_arr = np.atleast_1d(d_rate_dt)
            rate_scalar = float(rate_arr[0])
            d_rate_dt_scalar = float(d_rate_dt_arr[0])
            return rate_scalar + t_val * d_rate_dt_scalar
        t = np.maximum(t, 1e-6)
        t_arr = np.asarray(t, dtype=float)
        rate = np.asarray(splev(t_arr, self.yield_spl, ext=0), dtype=float)
        d_rate_dt = np.asarray(splev(t_arr, self.yield_spl, der=1, ext=0), dtype=float)
        return rate + t_arr * d_rate_dt

    # --------------------------------------------------------------------- #
    # Monte-Carlo pricing
    # --------------------------------------------------------------------- #

    def price_options(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        params: tuple[float, float, float] | np.ndarray,
        n_paths: int = 50000,
        n_steps: int = 512,
        return_iv: bool = False,
        option_type: str = "call",
        return_terminal_paths: bool = False,
        precomputed_g1: np.ndarray | None = None,
        precomputed_z: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Price a set of European options using Monte-Carlo under rBergomi dynamics.

        Parameters
        ----------
        strikes, maturities : np.ndarray
            Grid of strikes and maturities to price
        params : tuple/array
            (eta, H, rho)
        n_paths, n_steps : int
            Monte-Carlo resolution
        return_iv : bool
            Return implied volatilities instead of prices
        option_type : {'call', 'put'}
        return_terminal_paths : bool
            Return raw terminal stock levels (used internally by calibration)

        Returns
        -------
        np.ndarray
            Prices or IVs on the requested grid
        """
        eta, hurst, rho = params
        alpha = hurst - 0.5
        n_steps = (
            4096 if max(maturities) < 0.1 else n_steps
        )  # finer grid for short maturities

        max_t = max(maturities)
        dt = max_t / n_steps
        K = precompute_convolution_matrix(alpha, n_steps)

        t = np.linspace(0, max_t, n_steps + 1)
        xi = self.forward_variance_func(t)
        short_rates = np.asarray(self.get_short_rate(t), dtype=float)

        if np.any(xi <= 0) or np.any(np.isnan(xi)):
            self.logger.error(f"Invalid forward variance values: {xi}")
            raise ValueError("Forward variance contains non-positive or NaN values")
        self.logger.info(f"Forward variance shape: {xi.shape}, values: {xi[:5]}...")

        # --------------------- Antithetic variates --------------------- #
        # Optionally reuse precomputed normals for calibration stability
        if precomputed_g1 is None or precomputed_z is None:
            seed = self.cfg.seed
            rng = default_rng(seed)
            n_base = n_paths // 2

            all_g1_base = rng.standard_normal((n_base, n_steps))
            all_g1 = np.vstack([all_g1_base, -all_g1_base])[:n_paths, :]

            all_z_base = rng.standard_normal((n_base, n_steps))
            all_z = np.vstack([all_z_base, -all_z_base])[:n_paths, :]
        else:
            all_g1 = precomputed_g1
            all_z = precomputed_z

        # --------------------- Simulation --------------------- #
        s_paths, variance_paths = simulate_rbergomi_paths(
            n_paths, n_steps, max_t, eta, hurst, rho, xi, all_g1, all_z, short_rates, K
        )
        if np.any(np.isnan(s_paths)) or np.any(s_paths <= 0):
            self.logger.error("Stock paths contain NaN or non-positive values")
            raise ValueError("Invalid stock paths")
        if np.any(np.isnan(variance_paths)) or np.any(variance_paths <= 0):
            self.logger.error("Variance paths contain NaN or non-positive values")
            raise ValueError("Invalid variance paths")

        # --------------------- Pricing --------------------- #
        indices = np.round(maturities / dt).astype(np.int64)
        s_t_all = s_paths[:, indices]  # (n_paths, n_maturities)

        yields = np.asarray(self.get_yield(maturities), dtype=float)
        discounts = np.exp(-yields * maturities)

        s_t = s_t_all
        discounts = discounts[:, None]

        if return_terminal_paths:
            # Shape (n_paths, n_maturities, 1)
            return s_t[:, :, np.newaxis]

        # Payoffs on already-discounted paths
        if option_type == "call":
            payoffs = np.maximum(s_t[:, :, None] - strikes[None, None, :], 0)
        else:  # put
            payoffs = np.maximum(strikes[None, None, :] - s_t[:, :, None], 0)

        prices = np.mean(payoffs, axis=0) * discounts  # discount after the nonlinearity
        prices = np.clip(prices, 1e-12, None)

        if return_iv:
            ivs = np.zeros(prices.shape)
            for m, maturity in enumerate(maturities):
                r = float(yields[m])
                for k, strike in enumerate(strikes):
                    ivs[m, k] = (
                        implied_volatility(prices[m, k], 1.0, strike, maturity, r=r)
                        if prices[m, k] > 0
                        else np.nan
                    )
            return ivs
        return prices

    # --------------------------------------------------------------------- #
    # Calibration
    # --------------------------------------------------------------------- #

    def calibrate(
        self,
        target_call_prices: Optional[np.ndarray],
        target_put_prices: Optional[np.ndarray],
        strikes: np.ndarray,
        maturities: np.ndarray,
        initial_params: tuple[float, float, float] | np.ndarray,
    ) -> "CalibrationResult":
        """
        Global calibration of (η, H, ρ) to market call/put prices using least-squares.

        Uses a single Monte-Carlo simulation per parameter set (shared paths)
        and adds mild regularization to improve convergence.

        Parameters
        ----------
        target_call_prices, target_put_prices : Optional[np.ndarray]
            Market call and put option prices
        strikes, maturities : np.ndarray
            Strike and maturity grids
        initial_params: Initial guess for [eta, hurst, rho].

        Returns
        -------
            CalibrationResult object containing optimal parameters and fit metrics.
        """
        # Filter very short maturities (numerically unstable)
        valid_indices = maturities >= 0.01
        filtered_maturities = maturities[valid_indices]

        # Ensure target price arrays are present; if None, replace with NaN arrays
        n_maturities = np.asarray(maturities).size
        n_strikes = np.asarray(strikes).size
        if target_call_prices is None:
            target_call_prices = np.full((n_maturities, n_strikes), np.nan)
        if target_put_prices is None:
            target_put_prices = np.full((n_maturities, n_strikes), np.nan)

        filtered_call_prices = target_call_prices[valid_indices]
        filtered_put_prices = target_put_prices[valid_indices]
        filtered_strikes = strikes
        self.logger.info(f"Filtered to {len(filtered_maturities)} maturities >= 0.01")

        # ----------------------------------------------------------------- #
        # Objective: relative pricing errors + regularization
        # ----------------------------------------------------------------- #
        bounds = ([0.01, 0.001, -0.999], [2.0, 0.49, 0.95])
        # Pre-generate shared randomness for calibration to make the objective deterministic
        n_paths = self.cfg.calibration.n_paths
        n_steps = self.cfg.calibration.n_steps
        seed = self.cfg.seed

        rng_cal = default_rng(seed)
        n_base_cal = n_paths // 2
        g1_base_cal = rng_cal.standard_normal((n_base_cal, n_steps))
        all_g1_cal = np.vstack([g1_base_cal, -g1_base_cal])[:n_paths, :]
        z_base_cal = rng_cal.standard_normal((n_base_cal, n_steps))
        all_z_cal = np.vstack([z_base_cal, -z_base_cal])[:n_paths, :]

        def objective(params):
            eta, hurst, rho = params

            # One shared MC simulation for both calls and puts using precomputed normals
            s_t = self.price_options(
                filtered_strikes,
                filtered_maturities,
                params,
                n_paths=n_paths,
                n_steps=n_steps,
                return_terminal_paths=True,
                precomputed_g1=all_g1_cal,
                precomputed_z=all_z_cal,
            ).squeeze(-1)  # shape: (n_paths_cal, n_maturities)

            discounts = np.exp(
                -self.get_yield(filtered_maturities) * filtered_maturities
            )[:, None]

            call_payoffs = np.maximum(
                s_t[:, :, None] - filtered_strikes[None, None, :], 0
            )
            put_payoffs = np.maximum(
                filtered_strikes[None, None, :] - s_t[:, :, None], 0
            )

            model_call_prices = np.mean(call_payoffs, axis=0) * discounts
            model_put_prices = np.mean(put_payoffs, axis=0) * discounts

            valid_call_mask = (filtered_call_prices > 0) & np.isfinite(
                filtered_call_prices
            )
            valid_put_mask = (filtered_put_prices > 0) & np.isfinite(
                filtered_put_prices
            )

            # Relative errors only on quoted instruments
            call_errors = (
                model_call_prices[valid_call_mask]
                - filtered_call_prices[valid_call_mask]
            ) / filtered_call_prices[valid_call_mask]
            put_errors = (
                model_put_prices[valid_put_mask] - filtered_put_prices[valid_put_mask]
            ) / filtered_put_prices[valid_put_mask]

            # Light L2 regularization toward realistic values
            reg_weight = self.cfg.calibration.regularisation_weight
            param_penalty = reg_weight * sum(
                [(eta - 1.0) ** 2, 10 * (hurst - 0.07) ** 2, (rho + 0.6) ** 2]
            )

            all_errors = np.concatenate([call_errors, put_errors])
            if len(all_errors) > 2:
                smoothness = 0.0001 * np.sum(np.diff(all_errors, n=2) ** 2)
            else:
                smoothness = 0.0

            total_penalty = param_penalty + smoothness

            return np.concatenate([all_errors, [np.sqrt(total_penalty)]])

        res = least_squares(
            objective,
            np.asarray(initial_params),
            bounds=bounds,
            method="trf",
            max_nfev=5000,
            ftol=1e-12,
            gtol=1e-9,
            xtol=1e-12,
            loss="soft_l1",
            verbose=2,
        )

        optimal_params = res.x

        # Compute fitted IVs (using both call and put prices for maximum consistency)
        s_t_final = self.price_options(
            filtered_strikes,
            filtered_maturities,
            optimal_params,
            return_terminal_paths=True,
        ).squeeze(-1)  # (n_paths, n_maturities)

        discounts = np.exp(-self.get_yield(filtered_maturities) * filtered_maturities)[
            :, None
        ]  # (n_maturities, 1)

        call_payoffs = np.maximum(
            s_t_final[:, :, None] - filtered_strikes[None, None, :], 0
        )
        fitted_call_prices = np.mean(call_payoffs, axis=0) * discounts
        fitted_ivs = np.full_like(fitted_call_prices, np.nan)

        for m, maturity in enumerate(filtered_maturities):
            r = float(np.array(self.get_yield(maturity)).item())
            for k, strike in enumerate(filtered_strikes):
                price = fitted_call_prices[m, k]
                if price > 0 and np.isfinite(price):
                    fitted_ivs[m, k] = implied_volatility(
                        price, 1.0, strike, maturity, r=r, option_type="call"
                    )

        # Final high-precision pricing with optimal parameters
        result = CalibrationResult(self.cfg)
        result.optimal_params = {
            "eta": optimal_params[0],
            "hurst": optimal_params[1],
            "rho": optimal_params[2],
        }
        result.fitted_ivs = fitted_ivs
        result.market_ivs = self.iv_surface[valid_indices]

        # Ensure arrays are not None before using numpy ufuncs to satisfy static type checkers
        if result.fitted_ivs is None or result.market_ivs is None:
            result.rmse = np.nan
            self.logger.warning(
                "Cannot compute RMSE because fitted_ivs or market_ivs is None"
            )
        else:
            valid_mask = np.isfinite(result.fitted_ivs) & np.isfinite(result.market_ivs)
            if np.any(valid_mask):
                result.rmse = np.sqrt(
                    np.mean(
                        (result.fitted_ivs[valid_mask] - result.market_ivs[valid_mask])
                        ** 2
                    )
                )
            else:
                result.rmse = np.nan
                self.logger.warning("No valid IVs for RMSE calculation")

        result.convergence_info = res
        return result


# =============================================================================
# Calibration Result Container & Reporting
# =============================================================================


class CalibrationResult:
    """
    Stores and visualizes calibration results for the Rough Bergomi model.

    Attributes:
        optimal_params: Dictionary of calibrated parameters (eta, hurst, rho).
        fitted_ivs: Array of fitted implied volatilities.
        market_ivs: Array of market implied volatilities.
        rmse: Root mean square error of the calibration.
        convergence_info: Optimization results from least_squares.
        simulation_stats: Placeholder for simulation statistics.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.logger = get_logger("CalibrationResult")
        self.optimal_params: Dict[str, float] = {}
        self.fitted_ivs: Optional[np.ndarray] = None
        self.market_ivs: Optional[np.ndarray] = None
        self.rmse: float = 0.0
        self.convergence_info: OptimizeResult = OptimizeResult()
        self.simulation_stats = {}

        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        self.save_path = Path(repo_root / self.cfg.rbergomi.save_path).resolve()

    def plot_fit_quality(self) -> None:
        """
        Plot market vs. model implied volatilities.
        """
        if self.market_ivs is None or self.fitted_ivs is None:
            self.logger.warning("Cannot plot - IV surfaces not computed")
            return

        fig, ax = plt.subplots()
        _unused = fig
        ax.plot(self.market_ivs.flatten(), label="Market IV")
        ax.plot(self.fitted_ivs.flatten(), label="Model IV")
        ax.legend()
        ax.set_title("Market vs Model Implied Volatilities")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Implied Volatility")

        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_path / "market_vs_model_iv.svg"
        plt.savefig(self.save_path, format="svg")

    def generate_report(self) -> Dict:
        """
        Generate a dictionary report of calibration results.

        Returns:
            Dictionary containing calibration parameters, RMSE, and convergence info.
        """
        convergence = self.convergence_info

        # Basic convergence fields
        conv = {
            "success": convergence.success,
            "status": convergence.status,
            "message": convergence.message,
            "x": convergence.x.tolist(),
            "cost": convergence.cost,
            "grad": (
                convergence.grad.tolist()
                if hasattr(convergence.grad, "tolist")
                else list(convergence.grad)
            ),
            "optimality": convergence.optimality,
            "active_mask": (
                convergence.active_mask.tolist()
                if hasattr(convergence.active_mask, "tolist")
                else list(convergence.active_mask)
            ),
            "nfev": convergence.nfev,
            "njev": convergence.njev,
        }
        # Summarize fun and jac if available
        try:
            if hasattr(convergence, "fun") and convergence.fun is not None:
                fun = np.asarray(convergence.fun)
                conv["fun_summary"] = {
                    "count": int(fun.size),
                    "mean": float(np.nanmean(fun)) if fun.size > 0 else None,
                    "std": float(np.nanstd(fun)) if fun.size > 0 else None,
                    "min": float(np.nanmin(fun)) if fun.size > 0 else None,
                    "max": float(np.nanmax(fun)) if fun.size > 0 else None,
                    "norm": float(np.linalg.norm(fun)) if fun.size > 0 else None,
                    "sample_first_10": fun.flatten()[:10].tolist()
                    if fun.size > 0
                    else [],
                }
        except (ValueError, TypeError, np.linalg.LinAlgError):
            conv["fun_summary"] = None

        try:
            if hasattr(convergence, "jac") and convergence.jac is not None:
                jac = np.asarray(convergence.jac)
                jac_shape = tuple(jac.shape)
                jac_rank = int(np.linalg.matrix_rank(jac))
                jac_cond = float(np.linalg.cond(jac)) if jac.size > 0 else None
                s = np.linalg.svd(jac, compute_uv=False)
                top_s = s[:10].tolist()

                conv["jac_summary"] = {
                    "shape": jac_shape,
                    "rank": jac_rank,
                    "cond": jac_cond,
                    "top_singular_values": top_s,
                }
        except (ValueError, TypeError, np.linalg.LinAlgError):
            conv["jac_summary"] = None

        return {
            "Optimal Params": self.optimal_params,
            "RMSE": self.rmse,
            "Convergence": conv,
        }


# =============================================================================
# Hydra Entry Point
# =============================================================================


@hydra.main(version_base=None, config_path="../configs", config_name="rbergomi_model")
def main(cfg: DictConfig):
    """
    Main entry point for rbergomi model calculation.

    Initialises logging, creates a RoughBergomiEngine instance, and
    executes the full calculation pipeline.

    Args:
        cfg: Optional configuration object; defaults to None.
    """
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")
    logger.info("Starting rough bergomi model calculation...")
    rbergomi_model = RoughBergomiEngine(cfg)

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    maturities_path = Path(repo_root / cfg.iv_surface.maturities_path).resolve()

    maturities = np.load(maturities_path)
    initial_params = np.array([1.5, 0.05, -0.95])  # Initial guess for [eta, hurst, rho]
    logger.info(f"Initial parameters: {initial_params}")

    logger.info("Pricing options...")
    option_prices = rbergomi_model.price_options(
        strikes=rbergomi_model.strikes,
        maturities=maturities,
        params=initial_params,
        n_paths=100000,
        n_steps=2048,
        return_iv=False,
    )
    logger.info(f"Option prices:\n{option_prices}")

    logger.info("Calibrating model...")
    # Convert market IV surface to model prices
    target_call_prices = np.zeros_like(rbergomi_model.iv_surface)
    target_put_prices = np.zeros_like(rbergomi_model.iv_surface)

    for m, maturity in enumerate(maturities):
        r = float(np.asarray(rbergomi_model.get_yield(maturity)))
        for k, strike in enumerate(rbergomi_model.strikes):
            if (
                np.isnan(rbergomi_model.iv_surface[m, k])
                or rbergomi_model.iv_surface[m, k] <= 0
            ):
                target_call_prices[m, k] = np.nan
                target_put_prices[m, k] = np.nan
            else:
                sigma = rbergomi_model.iv_surface[m, k]
                target_call_prices[m, k] = black_scholes_call(
                    s=1.0, k=strike, t=maturity, r=r, q=0, sigma=sigma
                )
                target_put_prices[m, k] = black_scholes_put(
                    s=1.0, k=strike, t=maturity, r=r, q=0, sigma=sigma
                )
                # Verify put-call parity as a sanity check
                parity_diff = abs(
                    (target_call_prices[m, k] - target_put_prices[m, k]) 
                    - (1.0 - strike * np.exp(-r * maturity))
                )
                if parity_diff > 1e-6:
                    logger.warning(
                        "Put-call parity violation at K=%.3f, T=%.3f: diff=%.2e",
                        strike,
                        maturity,
                        parity_diff,
                    )

    calibration_result = rbergomi_model.calibrate(
        target_call_prices=target_call_prices,
        target_put_prices=target_put_prices,
        strikes=rbergomi_model.strikes,
        maturities=maturities,
        initial_params=initial_params,
    )

    logger.info("Generating calibration report...")
    report = calibration_result.generate_report()
    logger.info(f"Calibration report:\n{report}")

    report_file = calibration_result.save_path / "calibration_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=str)
    logger.info(f"Calibration report saved to {report_file}")

    logger.info("Plotting fit quality...")
    calibration_result.plot_fit_quality()
    logger.info(f"Plots saved to {calibration_result.save_path}")

    logger.info("Rough Bergomi model calculation completed.")


if __name__ == "__main__":
    main()
