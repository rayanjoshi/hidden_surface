"""Black-Scholes formulas and implied volatility utilities.

This module provides fast, vectorised implementations of:
- Call and put option pricing
- Vega (sensitivity to volatility)
- Implied volatility computation via Brent's method

All functions are robust to edge cases (zero time/volatility) and are safe to use
in Monte-Carlo pricing and calibration loops.

Functions
---------
compute_vega
    Vectorised Black-Scholes vega (∂C/∂σ), scaled to percentage terms.
compute_vega_scalar
    Scalar version used in tight calibration loops.
black_scholes_call
    Vectorised Black-Scholes call price.
black_scholes_put
    Vectorised Black-Scholes put price.
implied_volatility
    Compute implied volatility from market price using Brent's method.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def compute_vega(
    s: float | np.ndarray,
    k: float | np.ndarray,
    t: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
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
            d1 = (
                np.log(s_v / k_v)
                + (r + 0.5 * sigma[valid if np.ndim(sigma) > 0 else ...] ** 2)
                * t[valid if np.ndim(t) > 0 else ...]
            ) / (sigma[valid if np.ndim(sigma) > 0 else ...] * sqrt_t + 1e-10)
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
