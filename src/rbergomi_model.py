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
from typing import Optional
from pathlib import Path
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

def black_scholes_call(s, k, t, r=0, q=0, sigma = 0.2):
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
    r = np.asarray(r) if not isinstance(r, float) else r
    q = np.asarray(q) if not isinstance(q, float) else q
    sigma = np.asarray(sigma) if not isinstance(sigma, float) else sigma

    sigma = np.where(sigma < 1e-10, 1e-10, sigma)
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t + 1e-10)
    d2 = d1 - sigma * sqrt_t
    call_price = s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    call_price = np.where(sigma < 1e-10, np.maximum(s - k, 0.0) * np.exp(-r * t), call_price)
    return call_price

def implied_volatility_call(price, s, k, t, r=0, q=0, option_type='call'):
    """
    Compute implied volatility for a call option using Black-Scholes.

    Args:
        price: Observed option price.
        s: Current stock price.
        k: Strike price.
        t: Time to maturity in years.
        r: Risk-free rate (default: 0.0).
        q: Dividend yield (default: 0.0).
        option_type: Option type, currently only 'call' is supported (default: 'call').

    Returns:
        Implied volatility or np.nan if computation fails.
    """
    if price <= 0 or np.isnan(price):
        return np.nan
    if option_type == 'call':
        def objective(sigma):
            model_price = black_scholes_call(s, k, t, r, q, sigma)
            return model_price - price
        try:
            iv = brentq(objective, 0.0001, 20.0, maxiter=500)
            return iv
        except ValueError as e:
            msg = (
                f"Failed to compute IV for price={price}, s={s}, k={k}, t={t}: {e}"
            )
            get_logger("implied_volatility_call").warning(msg)
            iv = np.nan
            return iv

@njit
def get_power_law_coefficients(alpha, steps):
    """
    Calculate power-law coefficients for the Rough Bergomi kernel.

    Args:
        alpha: Roughness parameter (H - 0.5 where H is Hurst exponent).
        steps: Number of time steps.

    Returns:
        Array of power-law coefficients.
    """
    coefficients = np.zeros(steps)
    eps = 1e-10

    for i in range(steps):
        k = i + 1
        num = k ** (alpha + 1) - (k - 1) ** (alpha + 1)
        base = num / (alpha + 1)
        base = max(base, eps)
        coefficients[k - 1] = base ** (1.0 / alpha)
    return coefficients

@njit
def get_kernel_values(alpha, steps, time_step):
    """
    Compute kernel values for the Rough Bergomi model.

    Args:
        alpha: Roughness parameter (H - 0.5 where H is Hurst exponent).
        steps: Number of time steps.
        time_step: Size of each time step.

    Returns:
        Array of kernel values.
    """
    coefficients = get_power_law_coefficients(alpha, steps)
    kernel = np.zeros(steps)
    eps = 1e-10
    for l in range(1, steps):
        coeff = coefficients[l - 1]
        coeff = max(coeff, eps)
        kernel[l] = (coeff / time_step) ** alpha
    return kernel

@njit
def create_toeplitz(kernel):
    """
    Create a Toeplitz matrix from kernel values.

    Args:
        kernel: Array of kernel values.

    Returns:
        Toeplitz matrix.
    """
    n = len(kernel)
    toeplitz = np.zeros((n, n))
    for j in range(n):
        for i in range(j, n):
            toeplitz[i, j] = kernel[i - j]
    return toeplitz

@njit(parallel=True)
def simulate_rbergomi_paths(n_paths, n_steps, maturity, eta, hurst,
                            rho, f_variance, all_g1, all_z, all_g2, short_rates):
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
        all_g2: Additional random increments for variance convolution.
        short_rates: Short rates for each time step.

    Returns:
        Tuple of (stock paths, variance paths).
    """
    alpha = hurst - 0.5
    alpha = max(alpha, -0.45)
    dt = maturity / n_steps
    n_time = n_steps + 1
    kernel = get_kernel_values(alpha, n_time, dt)
    toeplitz = create_toeplitz(kernel)
    covariance_factor = dt ** (2 * alpha + 1) / (2 * alpha + 1)

    delta_w1 = np.zeros((n_paths, n_time))
    delta_w1[:, 1:] = all_g1 * np.sqrt(dt)
    delta_w2 = np.zeros((n_paths, n_time))
    delta_w2[:, 1:] = all_z * np.sqrt(dt)
    delta_ws = rho * delta_w1 + np.sqrt(1 - rho ** 2) * delta_w2

    recent_with_zero = np.zeros((n_paths, n_time))
    recent_with_zero[:, 1:] = np.sqrt(covariance_factor) * all_g2

    s_paths = np.zeros((n_paths, n_time))
    variance_paths = np.zeros((n_paths, n_time))
    s_paths[:, 0] = 1.0
    variance_paths[:, 0] = f_variance[0]

    w = np.zeros((n_paths, n_time))
    for p in prange(n_paths): # pylint: disable=not-an-iterable
        conv = np.dot(toeplitz, delta_w1[p])
        w[p] = conv + recent_with_zero[p]
    w[:, 0] = 0

    t = np.linspace(0, maturity, n_time)
    t_2h = t ** (2 * hurst)
    t_2h_tile = np.zeros((n_paths, n_time))
    for p in prange(n_paths): # pylint: disable=not-an-iterable
        t_2h_tile[p, :] = t_2h
    exponent = eta * np.sqrt(2 * hurst) * w - 0.5 * (eta ** 2) * t_2h_tile
    exponent = np.clip(exponent, -100, 100)
    f_variance_tile = np.zeros((n_paths, n_time))
    for p in prange(n_paths): # pylint: disable=not-an-iterable
        f_variance_tile[p, :] = f_variance
    variance_paths = f_variance_tile * np.exp(exponent)
    min_variance = 1e-10
    variance_paths = np.maximum(variance_paths, min_variance)

    # Vectorized stock path update with prange for cumsum
    sqrt_variance = np.sqrt(variance_paths[:, :-1])
    short_rates_tile = np.zeros((n_paths, n_steps))
    for p in prange(n_paths): # pylint: disable=not-an-iterable
        short_rates_tile[p, :] = short_rates[:-1]
    drift = (short_rates_tile - 0.5 * variance_paths[:, :-1]) * dt
    diffusion = sqrt_variance * delta_ws[:, 1:]
    d_log_s = drift + diffusion
    log_s_cum = np.zeros_like(d_log_s)
    for p in prange(n_paths): # pylint: disable=not-an-iterable
        log_s_cum[p] = np.cumsum(d_log_s[p])
    s_paths[:, 1:] = np.exp(log_s_cum)

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


    def get_yield(self, t):
        """
        Interpolate yield curve at a given time.

        Args:
            t: Time point for yield interpolation.

        Returns:
            Interpolated yield value.
        """
        return splev(t, self.yield_spl, ext=0)

    def get_short_rate(self, t):
        """
        Compute short rate using yield curve and its derivative.

        Args:
            t: Time point(s) for short rate calculation.

        Returns:
            Short rate(s) at the specified time(s).
        """
        if np.isscalar(t):
            if t <= 0:
                t = 1e-6
            rate = splev(t, self.yield_spl, ext=0)
            d_rate_dt = splev(t, self.yield_spl, der=1, ext=0)
            return rate + t * d_rate_dt
        t = np.maximum(t, 1e-6)
        rate = splev(t, self.yield_spl, ext=0)
        d_rate_dt = splev(t, self.yield_spl, der=1, ext=0)
        return rate + t * d_rate_dt

    def price_options(self, strikes, maturities, params,
                    n_paths=50000, n_steps=512, return_iv=False):
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

        all_g2_base = np.random.randn(n_base, n_steps)
        all_g2 = np.vstack([all_g2_base, -all_g2_base])[:n_paths, :]

        s_paths, variance_paths = simulate_rbergomi_paths(n_paths, n_steps, max_t,
                                                        eta, hurst, rho, xi, all_g1, all_z,
                                                        all_g2, short_rates)
        if np.any(np.isnan(s_paths)) or np.any(s_paths <= 0):
            self.logger.error("Stock paths contain NaN or non-positive values")
            raise ValueError("Invalid stock paths")
        if np.any(np.isnan(variance_paths)) or np.any(variance_paths <= 0):
            self.logger.error("Variance paths contain NaN or non-positive values")
            raise ValueError("Invalid variance paths")

        # Vectorized pricing
        indices = np.round(maturities / dt).astype(int)
        s_t_all = s_paths[:, indices]
        yields = self.get_yield(maturities)
        discounts = np.exp(-yields * maturities)
        payoffs = np.maximum(s_t_all[:, :, None] - strikes[None, None, :], 0)
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
                        ivs[m, k] = implied_volatility_call(
                            prices[m, k],
                            1.0,
                            strike,
                            maturity,
                            r=r,
                        )
            return ivs
        return prices

    def calibrate(self, target_prices, strikes, maturities, initial_params):
        """
        Calibrate the Rough Bergomi model to market prices.

        Args:
            target_prices: Array of market option prices.
            strikes: Array of strike prices.
            maturities: Array of maturities.
            initial_params: Initial guess for [eta, hurst, rho].

        Returns:
            CalibrationResult object containing optimal parameters and fit metrics.
        """
        valid_indices = maturities >= 0.01
        filtered_maturities = maturities[valid_indices]
        filtered_target_prices = target_prices[valid_indices]
        filtered_strikes = strikes
        self.logger.info(f"Filtered to {len(filtered_maturities)} maturities >= 0.01")
        def objective(params):
            eta, hurst, rho = params
            model_prices = self.price_options(filtered_strikes, filtered_maturities, params)
            if np.any(np.isnan(model_prices)) or np.any(model_prices <= 0):
                self.logger.warning(f"Invalid model prices for params {params}: {model_prices}")
                return np.full(filtered_target_prices.size, 1e6)

            valid_mask = (filtered_target_prices > 0) & np.isfinite(filtered_target_prices)
            if not np.any(valid_mask):
                return np.full(filtered_target_prices.size, 1e6)

            model_flat = model_prices[valid_mask]
            target_flat = filtered_target_prices[valid_mask]
            relative_errors = (model_flat - target_flat) / target_flat

            reg_weight = 0.001
            param_penalty = reg_weight * sum([
                (eta - 1.0)**2,
                10 * (hurst - 0.07)**2,
                (rho + 0.6)**2
            ])

            if len(relative_errors) > 2:
                smoothness = 0.0001 * np.sum(np.diff(relative_errors, n=2)**2)
            else:
                smoothness = 0.0

            total_penalty = param_penalty + smoothness

            return np.concatenate([relative_errors, [total_penalty]])

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

        # Compute fitted IVs
        fitted_prices = self.price_options(filtered_strikes, filtered_maturities, optimal_params)
        fitted_ivs = np.zeros(fitted_prices.shape)
        for m, maturity in enumerate(filtered_maturities):
            r = self.get_yield(maturity)
            for k, strike in enumerate(strikes):
                if fitted_prices[m, k] <= 0 or np.isnan(fitted_prices[m, k]):
                    fitted_ivs[m, k] = np.nan
                else:
                    fitted_ivs[m, k] = implied_volatility_call(
                        fitted_prices[m, k], 1.0, strike, maturity, r=r
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
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.optimal_params = {}
        self.fitted_ivs = None
        self.market_ivs = None
        self.rmse = 0.0
        self.convergence_info = {}
        self.simulation_stats = {}
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        self.save_path = Path(repo_root / self.cfg.rbergomi.save_path).resolve()

    def plot_fit_quality(self):
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
        self.save_path = self.save_path / "market_vs_model_iv.png"
        plt.savefig( self.save_path)

    def generate_report(self):
        """
        Generate a text report of calibration results.

        Returns:
            String containing calibration parameters, RMSE, and convergence info.
        """
        report_lines = [
            f"Optimal Params: {self.optimal_params}",
            f"RMSE: {self.rmse}",
            f"Convergence: {self.convergence_info}",
        ]
        report = "\n".join(report_lines)
        return report


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
    market_prices = np.zeros_like(rbergomi_model.iv_surface)
    for m, maturity in enumerate(maturities):
        r = rbergomi_model.get_yield(maturity)
        for k, strike in enumerate(rbergomi_model.strikes):
            if np.isnan(rbergomi_model.iv_surface[m, k]) or rbergomi_model.iv_surface[m, k] <= 0:
                market_prices[m, k] = np.nan
            else:
                market_prices[m, k] = black_scholes_call(
                    s=1.0, k=strike, t=maturity, r=r, q=0, sigma=rbergomi_model.iv_surface[m, k]
                )
    target_prices = market_prices
    calibration_result = rbergomi_model.calibrate(
        target_prices=target_prices,
        strikes=rbergomi_model.strikes,
        maturities=maturities,
        initial_params=initial_params
    )

    logger.info("Generating calibration report...")
    report = calibration_result.generate_report()
    logger.info(f"Calibration report:\n{report}")

    logger.info("Plotting fit quality...")
    calibration_result.plot_fit_quality()
    logger.info(f"Plots saved to {calibration_result.save_path}")

    logger.info("Rough Bergomi model calculation completed.")

if __name__ == "__main__":
    main()
