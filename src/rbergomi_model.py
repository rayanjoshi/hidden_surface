"""
Rough Bergomi (rBergomi) Model Implementation
=============================================

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
- Robust calibration using trust-region reflective least-squares with regularisation
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


def compute_vega(
    s: float | np.ndarray,
    k: float | np.ndarray,
    t: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray
) -> float | np.ndarray:
    """
    Compute Black-Scholes Vega = ∂Price/∂σ.
    
    Vega measures the sensitivity of option price to volatility.
    Used for weighting calibration points by their information content.
    
    Mathematical Formula:
        Vega = S * √T * φ(d₁) / 100
        
    where φ is the standard normal PDF and d₁ is the usual BS d1.
    The division by 100 scales Vega to percentage terms.
    
    Parameters
    ----------
    s : float or np.ndarray
        Spot price (normalized to 1 for log-moneyness grids)
    k : float or np.ndarray
        Strike price(s)
    t : float or np.ndarray
        Time to maturity in years
    r : float
        Risk-free rate
    sigma : float or np.ndarray
        Implied volatility
    
    Returns
    -------
    float or np.ndarray
        Vega value(s)
    """
    # Handle edge cases
    t = np.asarray(t)
    sigma = np.asarray(sigma)
    
    if np.any(t <= 0) or np.any(sigma <= 0):
        # Return zeros for invalid inputs
        result = np.zeros_like(t * sigma)
        valid = (t > 0) & (sigma > 0)
        if np.any(valid):
            s_v = np.asarray(s)
            k_v = np.asarray(k)
            sqrt_t = np.sqrt(t[valid] if np.ndim(t) > 0 else t)
            d1 = (np.log(s_v / k_v) + (r + 0.5 * sigma[valid if np.ndim(sigma) > 0 else ...]**2) * t[valid if np.ndim(t) > 0 else ...]) / (sigma[valid if np.ndim(sigma) > 0 else ...] * sqrt_t + 1e-10)
            result[valid] = s_v * sqrt_t * norm.pdf(d1) / 100.0
        return result
    
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t + 1e-10)
    vega = s * sqrt_t * norm.pdf(d1) / 100.0
    
    return vega


def compute_vega_scalar(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """
    Compute Black-Scholes Vega for scalar inputs.
    
    Faster version for single option evaluation used in calibration loops.
    
    Parameters
    ----------
    s, k, t, r, sigma : float
        Standard Black-Scholes inputs
    
    Returns
    -------
    float
        Vega value (scaled to percentage terms)
    """
    if t <= 0 or sigma <= 0:
        return 0.0
    
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    vega = s * sqrt_t * norm.pdf(d1) / 100.0
    
    return float(vega)


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
        - Global least-squares calibration with regularisation
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
                float(self.df["forward_variance"].iloc[-1]),
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
            8196 if max(maturities) < 0.1 else n_steps
        )  # finer grid for short maturities

        max_t = max(maturities)
        dt = max_t / n_steps
        K = precompute_convolution_matrix(alpha, n_steps)

        t = np.linspace(0, max_t, n_steps + 1)
        xi = self.forward_variance_func(t)
        short_rates = np.asarray(self.get_short_rate(t), dtype=float)

        if np.any(xi <= 0) or np.any(np.isnan(xi)):
            self.logger.error("Invalid forward variance values: %s", xi)
            raise ValueError("Forward variance contains non-positive or NaN values")
        self.logger.info(
            "Forward variance shape: %s, values: %s...",
            xi.shape,
            xi[:5],
        )

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

    def calibration(
        self,
        target_call_prices: Optional[np.ndarray],
        target_put_prices: Optional[np.ndarray],
        strikes: np.ndarray,
        maturities: np.ndarray,
        initial_params: tuple[float, float, float] | np.ndarray,
    ) -> "CalibrationResult":
        """
        Calibrate (η, H, ρ) to market IV surface using Vega-weighted least squares.
        
        Mathematically rigorous calibration that minimises:
        
            Σᵢⱼ Vegaᵢⱼ * [(IV_model - IV_market) / IV_market]²
        
        Subject to parameter bounds:
            η ∈ [0.1, 10.0]      - Vol-of-vol parameter
            H ∈ [0.005, 0.20]    - Hurst parameter (rough regime)
            ρ ∈ [-1.0, -0.10]    - Spot-vol correlation (negative for equity)
        
        Key improvements over previous implementation:
        1. Vega weighting: Points with higher Vega (more sensitive to vol) get more weight
        2. No arbitrary parameter penalties: Let bounds handle constraints
        3. No smoothness penalty on errors: Forward variance handles smoothness
        4. Pure IV-space calibration: More stable than price-space
        
        Parameters
        ----------
        target_call_prices, target_put_prices : Optional[np.ndarray]
            Market call and put option prices (used only for fallback)
        strikes, maturities : np.ndarray
            Strike and maturity grids
        initial_params : tuple or np.ndarray
            Initial guess for [eta, hurst, rho]

        Returns
        -------
        CalibrationResult
            Object containing optimal parameters, fitted IVs, and fit metrics
        """
        # Filter very short maturities (numerically unstable)
        maturities_filter = self.cfg.calibration.maturities_filter
        valid_indices = maturities >= maturities_filter
        filtered_maturities = maturities[valid_indices]
        filtered_strikes = strikes
        
        # Market IVs for this calibration
        market_ivs = self.iv_surface[valid_indices]
        
        # Compute yields for each maturity
        yields = np.asarray(self.get_yield(filtered_maturities), dtype=float)
        
        self.logger.info(
            "Calibrating to %d maturities × %d strikes = %d points",
            len(filtered_maturities), len(filtered_strikes),
            len(filtered_maturities) * len(filtered_strikes)
        )

        # ----------------------------------------------------------------- #
        # Compute Vega weights for all calibration points
        # ----------------------------------------------------------------- #
        vegas = np.zeros_like(market_ivs)
        for m, (T, r) in enumerate(zip(filtered_maturities, yields)):
            for k, K in enumerate(filtered_strikes):
                if np.isfinite(market_ivs[m, k]) and market_ivs[m, k] > 0:
                    # Use S=1.0 for normalized grid
                    vegas[m, k] = compute_vega_scalar(1.0, K, T, r, market_ivs[m, k])
        
        # Normalize Vegas to have mean = 1 for numerical stability
        vega_mask = vegas > 0
        if np.any(vega_mask):
            vega_mean = np.mean(vegas[vega_mask])
            if vega_mean > 0:
                vegas = vegas / vega_mean
            else:
                vegas = np.ones_like(vegas)
        else:
            self.logger.warning("No valid Vega weights; using uniform weighting")
            vegas = np.ones_like(market_ivs)
        
        n_valid = np.sum((market_ivs > 0) & np.isfinite(market_ivs) & (vegas > 0))
        self.logger.info("Number of valid calibration points: %d", n_valid)

        # ----------------------------------------------------------------- #
        # Pre-generate shared randomness for deterministic calibration
        # ----------------------------------------------------------------- #
        bounds = ([0.1, 0.005, -1.0], [10.0, 0.20, -0.01]) # η, H, ρ
        n_paths = self.cfg.calibration.n_paths
        n_steps = self.cfg.calibration.n_steps
        seed = self.cfg.seed

        rng_cal = default_rng(seed)
        n_base_cal = n_paths // 2
        g1_base_cal = rng_cal.standard_normal((n_base_cal, n_steps))
        all_g1_cal = np.vstack([g1_base_cal, -g1_base_cal])[:n_paths, :]
        z_base_cal = rng_cal.standard_normal((n_base_cal, n_steps))
        all_z_cal = np.vstack([z_base_cal, -z_base_cal])[:n_paths, :]

        # ----------------------------------------------------------------- #
        # Vega-weighted IV-space objective function
        # ----------------------------------------------------------------- #
        # Pre-compute the fixed calibration mask (must be constant across calls)
        fixed_calibration_mask = (market_ivs > 0) & np.isfinite(market_ivs) & (vegas > 0)
        n_calibration_points = np.sum(fixed_calibration_mask)
        self.logger.info("Using %d fixed calibration points", n_calibration_points)
        
        def objective(params):
            """
            Compute Vega-weighted relative IV errors.
            
            Returns vector of weighted errors for least_squares optimiser.
            The optimiser will minimize sum(errors²), so we return:
                sqrt(vega) * (IV_model - IV_market) / IV_market
            
            This gives us Vega-weighted relative IV errors when squared.
            
            IMPORTANT: Must return same-sized array for all parameter values!
            """
            eta, hurst, rho = params

            # Single MC simulation with shared randomness
            s_t = self.price_options(
                filtered_strikes,
                filtered_maturities,
                params,
                n_paths=n_paths,
                n_steps=n_steps,
                return_terminal_paths=True,
                precomputed_g1=all_g1_cal,
                precomputed_z=all_z_cal,
            ).squeeze(-1)  # shape: (n_paths, n_maturities)

            discounts = np.exp(-yields * filtered_maturities)[:, None]

            # Compute model call prices
            call_payoffs = np.maximum(
                s_t[:, :, None] - filtered_strikes[None, None, :], 0
            )
            model_call_prices = np.mean(call_payoffs, axis=0) * discounts

            # Convert model prices to implied volatilities
            model_ivs = np.full_like(model_call_prices, np.nan)
            for m, (T, r) in enumerate(zip(filtered_maturities, yields)):
                for k, K in enumerate(filtered_strikes):
                    price = model_call_prices[m, k]
                    if price > 1e-12:
                        try:
                            model_ivs[m, k] = implied_volatility(
                                price, 1.0, K, T, r=r, option_type='call'
                            )
                        except Exception:
                            model_ivs[m, k] = np.nan

            # Initialise residuals array with fixed size
            residuals = np.zeros(n_calibration_points)
            
            # Get values at calibration points
            market_ivs_valid = market_ivs[fixed_calibration_mask]
            model_ivs_valid = model_ivs[fixed_calibration_mask]
            vega_weights = np.sqrt(vegas[fixed_calibration_mask])
            
            # Handle invalid model IVs by assigning large residual
            valid_model = np.isfinite(model_ivs_valid) & (model_ivs_valid > 0)
            
            # Where model is valid: compute relative error
            residuals[valid_model] = vega_weights[valid_model] * (
                (model_ivs_valid[valid_model] - market_ivs_valid[valid_model]) 
                / market_ivs_valid[valid_model]
            )
            
            # Where model is invalid: assign large penalty (but not too large to cause overflow)
            residuals[~valid_model] = 10.0  # 1000% relative error
            
            return residuals

        # ----------------------------------------------------------------- #
        # Run optimisation
        # ----------------------------------------------------------------- #
        self.logger.info("Starting calibration with initial params: eta=%.3f, H=%.4f, rho=%.3f",
                        initial_params[0], initial_params[1], initial_params[2])
        
        res = least_squares(
            objective,
            np.asarray(initial_params),
            bounds=bounds,
            method="trf",
            max_nfev=5000,
            ftol=1e-6,
            gtol=1e-6,
            xtol=1e-8,
            loss="soft_l1",  # Robust to outliers
            verbose=2,
        )

        optimal_params = res.x
        self.logger.info(
            "Optimal parameters: η=%.4f, H=%.4f, ρ=%.4f",
            optimal_params[0], optimal_params[1], optimal_params[2]
        )

        # ----------------------------------------------------------------- #
        # Compute fitted IVs with optimal parameters
        # ----------------------------------------------------------------- #
        s_t_final = self.price_options(
            filtered_strikes,
            filtered_maturities,
            optimal_params,
            n_paths=self.cfg.final_pricing.n_paths,
            n_steps=self.cfg.final_pricing.n_steps,
            return_terminal_paths=True,
        ).squeeze(-1)

        discounts = np.exp(-yields * filtered_maturities)[:, None]

        call_payoffs = np.maximum(
            s_t_final[:, :, None] - filtered_strikes[None, None, :], 0
        )
        fitted_call_prices = np.mean(call_payoffs, axis=0) * discounts
        fitted_ivs = np.full_like(fitted_call_prices, np.nan)

        for m, maturity in enumerate(filtered_maturities):
            r = float(yields[m])
            for k, strike in enumerate(filtered_strikes):
                price = fitted_call_prices[m, k]
                if price > 0 and np.isfinite(price):
                    try:
                        fitted_ivs[m, k] = implied_volatility(
                            price, 1.0, strike, maturity, r=r, option_type="call"
                        )
                    except Exception:
                        fitted_ivs[m, k] = np.nan

        # ----------------------------------------------------------------- #
        # Build calibration result
        # ----------------------------------------------------------------- #
        result = CalibrationResult(self.cfg)
        result.optimal_params = {
            "eta": float(optimal_params[0]),
            "hurst": float(optimal_params[1]),
            "rho": float(optimal_params[2]),
        }
        result.fitted_ivs = fitted_ivs
        result.market_ivs = market_ivs

        # Compute RMSE (unweighted, for reporting)
        if result.fitted_ivs is not None and result.market_ivs is not None:
            valid_mask = np.isfinite(result.fitted_ivs) & np.isfinite(result.market_ivs)
            if np.any(valid_mask):
                result.rmse = float(np.sqrt(
                    np.mean(
                        (result.fitted_ivs[valid_mask] - result.market_ivs[valid_mask]) ** 2
                    )
                ))
                
                # Also compute weighted RMSE for comparison
                if np.any(valid_mask & (vegas > 0)):
                    weighted_sq_errors = vegas[valid_mask] * (
                        (result.fitted_ivs[valid_mask] - result.market_ivs[valid_mask]) / 
                        result.market_ivs[valid_mask]
                    ) ** 2
                    weighted_rmse = float(np.sqrt(np.mean(weighted_sq_errors)))
                    self.logger.info("Vega-weighted RMSE (relative): %.6f", weighted_rmse)
            else:
                result.rmse = np.nan
                self.logger.warning("No valid IVs for RMSE calculation")
        else:
            result.rmse = np.nan
            self.logger.warning("Cannot compute RMSE because fitted_ivs or market_ivs is None")

        self.logger.info("Calibration RMSE (absolute IV): %.6f", result.rmse)
        
        result.convergence_info = res
        return result


# =============================================================================
# Calibration Result Container & Reporting
# =============================================================================


class CalibrationResult:
    """
    Stores and visualises calibration results for the Rough Bergomi model.

    Attributes:
        optimal_params: Dictionary of calibration parameters (eta, hurst, rho).
        fitted_ivs: Array of fitted implied volatilities.
        market_ivs: Array of market implied volatilities.
        rmse: Root mean square error of the calibration.
        convergence_info: Optimisation results from least_squares.
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
        Saves plot and data to the configured save path.
        """
        if self.market_ivs is None or self.fitted_ivs is None:
            self.logger.warning("Cannot plot - IV surfaces not computed")
            return

        fig, ax = plt.subplots()
        ax.plot(self.market_ivs.flatten(), label="Market IV")
        ax.plot(self.fitted_ivs.flatten(), label="Model IV")
        ax.legend()
        ax.set_title("Market vs Model Implied Volatilities")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Implied Volatility")

        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_path / "market_vs_model_iv.svg"
        plt.savefig(self.save_path, format="svg")
        np.save(self.save_path.with_name("fitted_ivs.npy"), self.fitted_ivs)
        np.save(self.save_path.with_name("market_ivs.npy"), self.market_ivs)

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
            "RMSE_IV": round(self.rmse, 4),
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
    initial_params = np.array([3.5, 0.07, -0.8])  # Initial guess for [eta, hurst, rho]
    logger.info("Initial parameters: %s", initial_params)

    logger.info("Pricing options...")
    option_prices = rbergomi_model.price_options(
        strikes=rbergomi_model.strikes,
        maturities=maturities,
        params=initial_params,
        n_paths=100000,
        n_steps=2048,
        return_iv=False,
    )
    logger.info("Option prices:\n%s", option_prices)

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

    calibration_result = rbergomi_model.calibration(
        target_call_prices=target_call_prices,
        target_put_prices=target_put_prices,
        strikes=rbergomi_model.strikes,
        maturities=maturities,
        initial_params=initial_params,
    )

    logger.info("Generating calibration report...")
    report = calibration_result.generate_report()
    logger.info("Calibration report:\n%s", report)

    report_file = calibration_result.save_path / "calibration_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=str)
    logger.info("Calibration report saved to %s", report_file)

    logger.info("Plotting fit quality...")
    calibration_result.plot_fit_quality()
    logger.info("Plots saved to %s", calibration_result.save_path)

    logger.info("Rough Bergomi model calculation completed.")


if __name__ == "__main__":
    main()
