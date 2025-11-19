"""
Rough Bergomi model implementation for option pricing and calibration.

This module provides functionality for simulating stock price paths using the
Rough Bergomi model, pricing options, and calibrating model parameters to market
data. It includes numerical methods for stochastic volatility modeling and
supports parallel computation with Numba.

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
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import interp1d, splev, splrep
from scipy.optimize import least_squares
import hydra
from omegaconf import DictConfig
from numba import njit, prange
import matplotlib.pyplot as plt

from scripts.logging_config import get_logger, setup_logging

def black_scholes_call(s: np.ndarray | float, k: np.ndarray | float, t: np.ndarray | float, r: float = 0, q: float = 0, sigma: float | np.ndarray = 0.2) -> np.ndarray:
    """
    Calculate the Black-Scholes call option price.

    Args:
        s: Current stock price(s).
        k: Strike price(s).
        t: Time to maturity in years.
        r: Risk-free rate (default: 0.0).
        q: Dividend yield (default: 0.0).
        sigma: Volatility (default: 0.2).

    Returns:
        Array of call option prices.
    """
    s = np.asarray(s)
    k = np.asarray(k)
    t = np.asarray(t)
    sigma = np.where(sigma < 1e-10, 1e-10, sigma)
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t + 1e-10)
    d2 = d1 - sigma * sqrt_t
    call_price = s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    call_price = np.where(sigma < 1e-10, np.maximum(s - k, 0.0) * np.exp(-r * t), call_price)
    return call_price

def black_scholes_put(s: np.ndarray | float, k: np.ndarray | float, t: np.ndarray | float, r: float = 0, q: float = 0, sigma: float | np.ndarray = 0.2):
    """
    Calculate the Black-Scholes put option price.

    Args:
        s: Current stock price(s).
        k: Strike price(s).
        t: Time to maturity in years.
        r: Risk-free rate (default: 0.0).
        q: Dividend yield (default: 0.0).
        sigma: Volatility (default: 0.2).

    Returns:
        Array of put option prices.
    """
    s = np.asarray(s)
    k = np.asarray(k)
    t = np.asarray(t)
    sigma = np.where(sigma < 1e-10, 1e-10, sigma)
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t + 1e-10)
    d2 = d1 - sigma * sqrt_t
    put_price = k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1)
    put_price = np.where(sigma < 1e-10, np.maximum(k - s, 0.0) * np.exp(-r * t), put_price)
    return put_price

def implied_volatility(price: float, s: float, k: float, t: float, r: float = 0, q: float = 0, option_type: str = 'call') -> float:
    """
    Compute implied volatility for a call or put option using Black-Scholes.

    Args:
        price: Observed option price.
        s: Current stock price.
        k: Strike price.
        t: Time to maturity in years.
        r: Risk-free rate (default: 0.0).
        q: Dividend yield (default: 0.0).
        option_type: Option type, 'call' or 'put' (default: 'call').

    Returns:
        Implied volatility or np.nan if computation fails.
    """
    if price <= 0 or np.isnan(price):
        return np.nan

    def objective(sigma: float) -> float:
        if option_type == 'call':
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


@njit
def get_kernel_values(alpha: float, n_time: int):
    """
    Compute kernel values for the Rough Bergomi model.

    Args:
        alpha: Roughness parameter (H - 0.5 where H is Hurst exponent).
        n_time: Number of time steps.

    Returns:
        Array of kernel values.
    """
    kernel = np.zeros(n_time)
    # compute kernel entries properly and divide once (avoid in-loop slicing)
    for m in range(n_time):
        # formula: ((m+1)^(alpha+1) - m^(alpha+1)) / (alpha+1)
        kernel[m] = ((m + 1) ** (alpha + 1) - m ** (alpha + 1)) / (alpha + 1)
    return kernel

@njit(parallel=True)
def simulate_rbergomi_paths(n_paths: int, n_steps: int, maturity: float, eta: float, hurst: float,
                            rho: float, f_variance, all_g1, all_z, short_rates):
    """
    Simulate stock and variance paths using the Rough Bergomi model.

    Args:
        n_paths: Number of simulation paths.
        n_steps: Number of time steps.
        maturity: Time to maturity in years.
        eta: Volatility of variance.
        hurst: Hurst exponent.
        rho: Correlation between stock and variance processes.
        f_variance: Forward variance curve.
        all_g1: Random increments for variance process.
        all_z: Random increments for stock process.
        short_rates: Short rates for each time step.

    Returns:
        Tuple of (stock paths, variance paths).
    """
    alpha = hurst - 0.5
    alpha = max(alpha, -0.45)
    dt = maturity / n_steps
    n_time = n_steps + 1
    kernel = get_kernel_values(alpha, n_time)

    delta_w1 = np.zeros((n_paths, n_time))
    delta_w1[:, 1:] = all_g1 * np.sqrt(dt)
    delta_w2 = np.zeros((n_paths, n_time))
    delta_w2[:, 1:] = all_z * np.sqrt(dt)
    delta_ws = rho * delta_w1 + np.sqrt(1 - rho ** 2) * delta_w2

    s_paths = np.zeros((n_paths, n_time))
    variance_paths = np.zeros((n_paths, n_time))
    s_paths[:, 0] = 1.0
    variance_paths[:, 0] = f_variance[0]

    w = np.zeros((n_paths, n_time))
    for p in prange(n_paths):
        for i in range(n_time):
            conv = 0.0
            # accumulate convolution using kernel
            for j in range(i + 1):
                conv += kernel[i - j] * delta_w1[p, j]
            w[p, i] = conv
    w = w * (dt ** alpha)   # scale w by dt**alpha

    # create time grid without using np.linspace to ensure numba compatibility
    t = np.empty(n_time)
    for i in range(n_time):
        t[i] = i * dt
    t_2h = t ** (2 * hurst)

    t_2h_tile = np.zeros((n_paths, n_time))
    for p in prange(n_paths):
        for i in range(n_time):
            t_2h_tile[p, i] = t_2h[i]

    exponent = eta * np.sqrt(2 * hurst) * w - 0.5 * (eta ** 2) * t_2h_tile
    # clip can be safely simulated via min/max in numba
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

    variance_paths = f_variance_tile * np.exp(exponent)
    min_variance = 1e-10
    for p in prange(n_paths):
        for i in range(n_time):
            if variance_paths[p, i] < min_variance:
                variance_paths[p, i] = min_variance

    # Vectorized stock path update with prange for cumsum
    sqrt_variance = np.sqrt(variance_paths[:, :-1])
    short_rates_tile = np.zeros((n_paths, n_steps))
    for p in prange(n_paths):
        for i in range(n_steps):
            short_rates_tile[p, i] = short_rates[i]

    drift = (short_rates_tile - 0.5 * variance_paths[:, :-1]) * dt
    diffusion = sqrt_variance * delta_ws[:, 1:]
    d_log_s = drift + diffusion

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

class RoughBergomiEngine:
    """
    Engine for pricing options and calibrating the Rough Bergomi model.

    Attributes:
        cfg: Configuration object containing file paths and model parameters.
        logger: Logger instance for tracking operations.
        df: DataFrame containing forward variance data.
        forward_variance_func: Interpolation function for forward variance.
        iv_surface: Implied volatility surface data.
        maturities: Array of option maturities.
        log_moneyness: Array of log moneyness values.
        strikes: Array of strike prices.
        risk_free_rate: Array of risk-free rates.
        yield_spl: Spline interpolation for risk-free rates.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.logger = get_logger("RoughBergomiEngine")
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        f_variance_path = Path(repo_root / self.cfg.f_variance.output_path).resolve()
        self.df = pd.read_csv(f_variance_path)
        self.forward_variance_func = interp1d(self.df["maturity_years"],
                                                self.df["forward_variance"],
                                                bounds_error=False,
                                                fill_value="extrapolate",
                                                )
        iv_surface_path = Path(repo_root / self.cfg.iv_surface.surface_path).resolve()
        self.iv_surface = np.load(iv_surface_path)
        if np.any(np.isnan(self.iv_surface)) or np.any(self.iv_surface <= 0):
            raise ValueError("IV surface contains NaN or non-positive values")

        maturities_path = Path(repo_root / self.cfg.iv_surface.maturities_path).resolve()
        self.maturities = np.load(maturities_path)
        log_moneyness_path = Path(repo_root / self.cfg.iv_surface.log_moneyness_path).resolve()
        self.log_moneyness = np.load(log_moneyness_path)
        if np.any(np.isnan(self.log_moneyness)):
            raise ValueError("Log moneyness contain NaN values")
        self.strikes = np.exp(self.log_moneyness)
        rf_path = self.cfg.iv_surface.risk_free_rate_path
        self.risk_free_rate_path = Path(repo_root / rf_path).resolve()
        self.risk_free_rate = np.load(self.risk_free_rate_path)

        if np.any(np.isnan(self.maturities)) or np.any(self.maturities <= 0):
            raise ValueError("Maturities contain NaN or non-positive values")

        risk_free_rates = self.risk_free_rate.copy()
        if self.maturities[0] > 0:
            self.maturities = np.insert(self.maturities, 0, 0)
            risk_free_rates = np.insert(self.risk_free_rate, 0, self.risk_free_rate[0])
        self.yield_spl = splrep(self.maturities, risk_free_rates, s=0, k=3)

        if self.df["maturity_years"].isnull().any() or self.df["forward_variance"].isnull().any():
            raise ValueError("Forward variance data contains NaN values")
        if (self.df["forward_variance"] <= 0).any():
            raise ValueError("Forward variance contains non-positive values")

        start = self.df["maturity_years"].min()
        end = self.df["maturity_years"].max()
        self.logger.info("Forward variance data range: %s to %s", start, end)


    def get_yield(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Interpolate yield curve at a given time.

        Args:
            t: Time point for yield interpolation.

        Returns:
            Interpolated yield value.
        """
        res = splev(t, self.yield_spl, ext=0)
        res_arr = np.asarray(res)
        if np.ndim(res_arr) == 0:
            return float(res_arr)
        return res_arr

    def get_short_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Compute short rate using yield curve and its derivative.

        Args:
            t: Time point(s) for short rate calculation.

        Returns:
            Short rate(s) at the specified time(s).
        """
        if np.isscalar(t):
            try:
                t_val = float(t)
            except (TypeError, ValueError):
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

    def price_options(self, strikes: np.ndarray, maturities: np.ndarray, params: tuple,
                    n_paths: int = 50000, n_steps: int = 512, return_iv: bool = False, option_type: str = 'call') -> np.ndarray:
        """
        Price options using the Rough Bergomi model.

        Args:
            strikes: Array of strike prices.
            maturities: Array of maturities.
            params: Model parameters [eta, hurst, rho].
            n_paths: Number of simulation paths (default: 50000).
            n_steps: Number of time steps (default: 512).
            return_iv: If True, return implied volatilities instead of prices (default: False).

        Returns:
            Array of option prices or implied volatilities.

        Raises:
            ValueError: If stock or variance paths contain invalid values.
        """
        eta, hurst, rho = params
        n_steps = 2048 if max(maturities) < 0.1 else n_steps
        max_t = max(maturities)
        dt = max_t / n_steps
        t = np.linspace(0, max_t, n_steps + 1)
        xi = self.forward_variance_func(t)
        short_rates = self.get_short_rate(t)
        if np.any(xi <= 0) or np.any(np.isnan(xi)):
            self.logger.error(f"Invalid forward variance values: {xi}")
            raise ValueError("Forward variance contains non-positive or NaN values")
        self.logger.info(f"Forward variance shape: {xi.shape}, values: {xi[:5]}...")

        # Antithetic variates for variance reduction
        np.random.seed(42)
        n_base = n_paths // 2

        all_g1_base = np.random.randn(n_base, n_steps)
        all_g1 = np.vstack([all_g1_base, -all_g1_base])[:n_paths, :]

        all_z_base = np.random.randn(n_base, n_steps)
        all_z = np.vstack([all_z_base, -all_z_base])[:n_paths, :]

        s_paths, variance_paths = simulate_rbergomi_paths(n_paths, n_steps, max_t,
                                                        eta, hurst, rho, xi, all_g1, all_z,
                                                        short_rates)
        if np.any(np.isnan(s_paths)) or np.any(s_paths <= 0):
            self.logger.error("Stock paths contain NaN or non-positive values")
            raise ValueError("Invalid stock paths")
        if np.any(np.isnan(variance_paths)) or np.any(variance_paths <= 0):
            self.logger.error("Variance paths contain NaN or non-positive values")
            raise ValueError("Invalid variance paths")

        # Vectorized pricing
        indices = np.round(maturities / dt).astype(int)
        s_t_all = s_paths[:, indices]
        yields = np.atleast_1d(self.get_yield(maturities))
        discounts = np.exp(-yields * maturities)
        if option_type == 'call':
            payoffs = np.maximum(s_t_all[:, :, None] - strikes[None, None, :], 0)
        else:  # 'put'
            payoffs = np.maximum(strikes[None, None, :] - s_t_all[:, :, None], 0)
        prices = np.mean(payoffs, axis=0) * discounts[:, None]
        prices = np.maximum(prices, 1e-12)

        if return_iv:
            ivs = np.zeros(prices.shape)
            for m, maturity in enumerate(maturities):
                r = yields[m]
                for k, strike in enumerate(strikes):
                    if prices[m, k] <= 0 or np.isnan(prices[m, k]):
                        self.logger.warning(
                            "Invalid price at maturity %s, strike %s: %s",
                            maturity,
                            strike,
                            prices[m, k],
                        )
                        ivs[m, k] = np.nan
                    else:
                        ivs[m, k] = implied_volatility(
                            prices[m, k],
                            1.0,
                            strike,
                            maturity,
                            r=r,
                        )
            return ivs
        return prices

    def calibrate(self, target_call_prices: Optional[np.ndarray], target_put_prices: Optional[np.ndarray], strikes: np.ndarray, maturities: np.ndarray, initial_params: tuple) -> 'CalibrationResult':
        """
        Calibrate the Rough Bergomi model to market prices.

        Args:
            target_call_prices: Array of market call option prices.
            target_put_prices: Array of market put option prices.
            strikes: Array of strike prices.
            maturities: Array of maturities.
            initial_params: Initial guess for [eta, hurst, rho].

        Returns:
            CalibrationResult object containing optimal parameters and fit metrics.
        """
        valid_indices = maturities >= 0.01
        filtered_maturities = maturities[valid_indices]
        filtered_call_prices = target_call_prices[valid_indices]
        filtered_put_prices = target_put_prices[valid_indices]
        filtered_strikes = strikes
        self.logger.info(f"Filtered to {len(filtered_maturities)} maturities >= 0.01")
        def objective(params):
            eta, hurst, rho = params
            model_call_prices = self.price_options(filtered_strikes,
                                                    filtered_maturities,
                                                    params,
                                                    option_type='call',
                                                    )
            model_put_prices = self.price_options(filtered_strikes,
                                                    filtered_maturities,
                                                    params,
                                                    option_type='put',
                                                    )

            valid_call_mask = (filtered_call_prices > 0) & np.isfinite(filtered_call_prices)
            valid_put_mask = (filtered_put_prices > 0) & np.isfinite(filtered_put_prices)

            call_errors = (
                model_call_prices[valid_call_mask] - filtered_call_prices[valid_call_mask]
            ) / filtered_call_prices[valid_call_mask]
            put_errors = (
                model_put_prices[valid_put_mask] - filtered_put_prices[valid_put_mask]
            ) / filtered_put_prices[valid_put_mask]

            reg_weight = 0.001
            param_penalty = reg_weight * sum([
                (eta - 1.0)**2,
                10 * (hurst - 0.07)**2,
                (rho + 0.6)**2
            ])

            all_errors = np.concatenate([call_errors, put_errors])
            if len(all_errors) > 2:
                smoothness = 0.0001 * np.sum(np.diff(all_errors, n=2)**2)
            else:
                smoothness = 0.0

            total_penalty = param_penalty + smoothness

            return np.concatenate([all_errors, [total_penalty]])

        bounds = ([0.01, 0.001, -0.95], [2.0, 0.49, 0.95])
        res = least_squares(
            objective,
            initial_params,
            bounds=bounds,
            method='trf',
            max_nfev=5000,
            ftol=1e-10,
            gtol=1e-10,
            xtol=1e-10,
            loss='soft_l1',
            verbose=2
        )

        optimal_params = res.x

        # Compute fitted IVs (using both call and put prices for maximum consistency)
        fitted_call_prices = self.price_options(filtered_strikes,
                                                filtered_maturities,
                                                optimal_params,
                                                option_type='call',
                                                )
        fitted_put_prices = self.price_options(filtered_strikes,
                                                filtered_maturities,
                                                optimal_params,
                                                option_type='put',
                                                )
        fitted_call_ivs = np.zeros(fitted_call_prices.shape)
        fitted_put_ivs = np.zeros(fitted_put_prices.shape)
        for m, maturity in enumerate(filtered_maturities):
            r = self.get_yield(maturity)
            for k, strike in enumerate(strikes):
                # Compute IV from calls
                if fitted_call_prices[m, k] > 0 and not np.isnan(fitted_call_prices[m, k]):
                    fitted_call_ivs[m, k] = implied_volatility(
                        fitted_call_prices[m, k], 1.0, strike, maturity, r=r, option_type='call'
                    )
                else:
                    fitted_call_ivs[m, k] = np.nan

                # Compute IV from puts
                if fitted_put_prices[m, k] > 0 and not np.isnan(fitted_put_prices[m, k]):
                    fitted_put_ivs[m, k] = implied_volatility(
                        fitted_put_prices[m, k], 1.0, strike, maturity, r=r, option_type='put'
                    )
                else:
                    fitted_put_ivs[m, k] = np.nan

        # Average the IVs from calls and puts (they should be identical due to put-call parity)
        fitted_ivs = np.zeros_like(fitted_call_ivs)
        valid_mask = np.isfinite(fitted_call_ivs) & np.isfinite(fitted_put_ivs)
        fitted_ivs[valid_mask] = (fitted_call_ivs[valid_mask] + fitted_put_ivs[valid_mask]) / 2
        mask = ~valid_mask
        fitted_ivs[mask] = np.where(
            np.isfinite(fitted_call_ivs[mask]),
            fitted_call_ivs[mask],
            fitted_put_ivs[mask]
        )

        result = CalibrationResult(self.cfg)
        result.optimal_params = {
            'eta': optimal_params[0],
            'hurst': optimal_params[1],
            'rho': optimal_params[2],
        }
        result.fitted_ivs = fitted_ivs
        result.market_ivs = self.iv_surface[valid_indices]

        valid_mask = np.isfinite(fitted_ivs) & np.isfinite(result.market_ivs)
        if np.any(valid_mask):
            result.rmse = np.sqrt(
                np.mean((fitted_ivs[valid_mask] - result.market_ivs[valid_mask]) ** 2)
            )
        else:
            result.rmse = np.nan
            self.logger.warning("No valid IVs for RMSE calculation")

        result.convergence_info = res
        return result


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
        self.optimal_params: Dict[str, float] = {}
        self.fitted_ivs: Optional[np.ndarray] = None
        self.market_ivs: Optional[np.ndarray] = None
        self.rmse: float = 0.0
        self.convergence_info = {}
        self.simulation_stats = {}
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        self.save_path = Path(repo_root / self.cfg.rbergomi.save_path).resolve()

    def plot_fit_quality(self) -> None:
        """
        Plot market vs. model implied volatilities.
        """
        fig, ax = plt.subplots()
        _unused = fig
        ax.plot(self.market_ivs.flatten(), label='Market IV')
        ax.plot(self.fitted_ivs.flatten(), label='Model IV')
        ax.legend()
        ax.set_title('Market vs Model Implied Volatilities')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Implied Volatility')
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_path / "market_vs_model_iv.svg"
        plt.savefig( self.save_path, format='svg')

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
            if hasattr(convergence.grad, 'tolist')
            else list(convergence.grad)
            ),
            "optimality": convergence.optimality,
            "active_mask": (
            convergence.active_mask.tolist()
            if hasattr(convergence.active_mask, 'tolist')
            else list(convergence.active_mask)
            ),
            "nfev": convergence.nfev,
            "njev": convergence.njev,
        }
        # Summarize fun and jac if available
        try:
            if hasattr(convergence, 'fun') and convergence.fun is not None:
                fun = np.asarray(convergence.fun)
                conv["fun_summary"] = {
                    "count": int(fun.size),
                    "mean": float(np.nanmean(fun)) if fun.size > 0 else None,
                    "std": float(np.nanstd(fun)) if fun.size > 0 else None,
                    "min": float(np.nanmin(fun)) if fun.size > 0 else None,
                    "max": float(np.nanmax(fun)) if fun.size > 0 else None,
                    "norm": float(np.linalg.norm(fun)) if fun.size > 0 else None,
                    "sample_first_10": fun.flatten()[:10].tolist() if fun.size > 0 else [],
                }
        except (ValueError, TypeError, np.linalg.LinAlgError):
            conv["fun_summary"] = None

        try:
            if hasattr(convergence, 'jac') and convergence.jac is not None:
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


@hydra.main(version_base=None, config_path="../configs", config_name="rbergomi_model")
def main(cfg: Optional[DictConfig] = None):
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
    initial_params = np.array([1.2, 0.07, -0.6])  # Initial guess for [eta, hurst, rho]
    logger.info(f"Initial parameters: {initial_params}")

    logger.info("Pricing options...")
    option_prices = rbergomi_model.price_options(
        strikes=rbergomi_model.strikes,
        maturities=maturities,
        params=initial_params,
        n_paths=50000,
        n_steps=1024,
        return_iv=False
    )
    logger.info(f"Option prices:\n{option_prices}")

    logger.info("Calibrating model...")
    target_call_prices = np.zeros_like(rbergomi_model.iv_surface)
    target_put_prices = np.zeros_like(rbergomi_model.iv_surface)
    for m, maturity in enumerate(maturities):
        r = rbergomi_model.get_yield(maturity)
        for k, strike in enumerate(rbergomi_model.strikes):
            if np.isnan(rbergomi_model.iv_surface[m, k]) or rbergomi_model.iv_surface[m, k] <= 0:
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
    calibration_result = rbergomi_model.calibrate(
        target_call_prices=target_call_prices,
        target_put_prices=target_put_prices,
        strikes=rbergomi_model.strikes,
        maturities=maturities,
        initial_params=initial_params
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
