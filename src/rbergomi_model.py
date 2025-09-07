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

from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end


def black_scholes_call(s, k, t, r=0, q=0, sigma = 0.2):
    log_function_start("black_scholes_call")
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    result = s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    log_function_end("black_scholes_call")
    return result

def implied_volatility_call(price, s, k, t, r=0, q=0, option_type='call'):
    log_function_start("implied_volatility_call")
    if price <= 0 or np.isnan(price):
        log_function_end("implied_volatility_call")
        return np.nan
    if option_type == 'call':
        def objective(sigma):
            model_price = black_scholes_call(s, k, t, r, q, sigma)
            return model_price - price
        try:
            iv = brentq(objective, 0.001, 20.0, maxiter=500)
            log_function_end("implied_volatility_call")
            return iv
        except ValueError as e:
            log_function_end("implied_volatility_call")
            get_logger("implied_volatility_call").warning(f"Failed to compute IV for price={price}, s={s}, k={k}, t={t}: {e}")
            iv = 0.001
            return iv

@njit
def get_power_law_coefficients(alpha, steps):
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
    coefficients = get_power_law_coefficients(alpha, steps)
    kernel = np.zeros(steps)
    eps = 1e-10
    for l in range(1, steps):
        coeff = coefficients[l - 1]
        coeff = max(coeff, eps)
        kernel[l] = (coeff / time_step) ** alpha
    return kernel

@njit
def compute_convolution(kernel, delta_w1):
    n = len(kernel)
    m = len(delta_w1)
    result = np.zeros(n + m - 1)
    for i in range(n + m - 1):
        for j in range(max(0, i - m + 1), min(n, i + 1)):
            if i - j < m:
                result[i] += kernel[j] * delta_w1[i - j]
    return result[:len(delta_w1)]

@njit(parallel=True)
def simulate_rbergomi_paths(n_paths, n_steps, maturity, eta, hurst,
                            rho, f_variance, all_g1, all_z, all_g2, short_rates):
    alpha = hurst - 0.5
    alpha = max(alpha, -0.45)
    dt = maturity / n_steps
    kernel = get_kernel_values(alpha, (n_steps + 1), dt)
    s_paths = np.zeros((n_paths, n_steps + 1))
    variance_paths = np.zeros((n_paths, n_steps + 1))
    t = np.linspace(0, maturity, n_steps + 1)
    s_paths[:, 0] = 1.0
    min_variance = 1e-10
    for p in prange(n_paths):  # pylint: disable=not-an-iterable
        variance_paths[p, 0] = f_variance[0]
        delta_w1 = np.zeros(n_steps + 1)
        delta_w1[1:] = all_g1[p] * np.sqrt(dt)
        covariance_factor = dt ** (2 * alpha + 1) / (2 * alpha + 1)
        recent = np.sqrt(covariance_factor) * all_g2[p]
        recent_with_zero = np.zeros(len(recent) + 1)
        recent_with_zero[1:] = recent
        conv = compute_convolution(kernel, delta_w1)
        w = conv + recent_with_zero
        w[0] = 0
        delta_w2 = np.zeros(n_steps + 1)
        delta_w2[1:] = np.sqrt(dt) * all_z[p]
        delta_ws = rho * delta_w1 + np.sqrt(1 - rho ** 2) * delta_w2
        pow_term = t ** (2 * hurst)

        exponent = eta * np.sqrt(2 * hurst) * w - 0.5 * (eta ** 2) * pow_term
        exponent = np.clip(exponent, -100, 100)
        variance = f_variance * np.exp(exponent)
        variance = np.maximum(variance, min_variance)
        variance_paths[p] = variance

        s = 1.0
        for i in range(1, n_steps + 1):
            if variance[i - 1] <= 0:
                variance[i - 1] = min_variance
            sqrt_variance = np.sqrt(variance[i - 1])
            d_log_s = (short_rates[i - 1] - 0.5 * variance[i - 1]) * dt + sqrt_variance * delta_ws[i]
            s = s * np.exp(d_log_s)
            s_paths[p, i] = s
    return s_paths, variance_paths

class RoughBergomiEngine:
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
        self.risk_free_rate_path = Path(repo_root / self.cfg.iv_surface.risk_free_rate_path).resolve()
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

        self.logger.info(f"Forward variance data range: {min(self.df['maturity_years'])} to {max(self.df['maturity_years'])}")


    def get_yield(self, t):
        return splev(t, self.yield_spl, ext=0)

    def get_short_rate(self, t):
        if np.isscalar(t):
            if t <= 0:
                t = 1e-6
            R = splev(t, self.yield_spl, ext=0)
            dRdt = splev(t, self.yield_spl, der=1, ext=0)
            return R + t * dRdt
        else:
            t = np.maximum(t, 1e-6)
            R = splev(t, self.yield_spl, ext=0)
            dRdt = splev(t, self.yield_spl, der=1, ext=0)
            return R + t * dRdt

    def price_options(self, strikes, maturities, params,
                        n_paths=50000, n_steps=512, return_iv=False):
        eta, hurst, rho = params
        max_t = max(maturities)
        dt = max_t / n_steps
        t = np.linspace(0, max_t, n_steps + 1)
        xi = self.forward_variance_func(t)
        short_rates = self.get_short_rate(t)
        if np.any(xi <= 0) or np.any(np.isnan(xi)):
            self.logger.error(f"Invalid forward variance values: {xi}")
            raise ValueError("Forward variance contains non-positive or NaN values")
        self.logger.info(f"Forward variance shape: {xi.shape}, values: {xi[:5]}...")

        np.random.seed(42)
        all_g1 = np.random.randn(n_paths // 2, n_steps)
        all_g1 = np.vstack([all_g1, -all_g1])
        all_z = np.random.randn(n_paths // 2, n_steps)
        all_z = np.vstack([all_z, -all_z])
        all_g2 = np.random.randn(n_paths // 2, n_steps)
        all_g2 = np.vstack([all_g2, -all_g2])

        if all_g1.shape[0] != n_paths:
            all_g1 = all_g1[:n_paths, :]
            all_z = all_z[:n_paths, :]
            all_g2 = all_g2[:n_paths, :]

        s_paths, variance_paths = simulate_rbergomi_paths(n_paths, n_steps, max_t,
                                                eta, hurst, rho, xi, all_g1, all_z,
                                                all_g2, short_rates)
        if np.any(np.isnan(s_paths)) or np.any(s_paths <= 0):
            self.logger.error("Stock paths contain NaN or non-positive values")
            raise ValueError("Invalid stock paths")
        if np.any(np.isnan(variance_paths)) or np.any(variance_paths <= 0):
            self.logger.error("Variance paths contain NaN or non-positive values")
            raise ValueError("Invalid variance paths")
        prices = np.zeros((len(maturities), len(strikes)))
        for m, maturity in enumerate(maturities):
            index = int(maturity / dt)
            s_t = s_paths[:, index]
            r = self.get_yield(maturity)
            discount = np.exp(-r * maturity)
            for k, strike in enumerate(strikes):
                payoff = np.maximum(s_t - strike, 0)
                discounted = np.mean(payoff) * discount
                prices[m, k] = discounted if discounted > 1e-12 else 1e-12
        if return_iv:
            ivs = np.zeros(prices.shape)
            for m, maturity in enumerate(maturities):
                r = self.get_yield(maturity)
                for k, strike in enumerate(strikes):
                    if prices[m, k] <= 0 or np.isnan(prices[m, k]):
                        self.logger.warning(f"Invalid price at maturity {maturity}, strike {strike}: {prices[m, k]}")
                        ivs[m, k] = np.nan
                    else:
                        ivs[m, k] = implied_volatility_call(prices[m, k], 1.0, strike, maturity, r=r)
            return ivs
        return prices

    def calibrate(self, target_prices, strikes, maturities, initial_params):
        valid_indices = maturities >= 0.01
        filtered_maturities = maturities[valid_indices]
        filtered_target_prices = target_prices[valid_indices]
        filtered_strikes = strikes
        self.logger.info(f"Filtered to {len(filtered_maturities)} maturities >= 0.01")
        def objective(params):
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

            penalty = 0.001 * np.sum([(p - ip)**2 for p, ip in zip(params, initial_params)])
            penalty_scaled = penalty * np.sqrt(len(relative_errors))
            return np.concatenate([relative_errors, np.array([penalty_scaled])])

        bounds = ([0.05, 0.005, -0.99], [2.0, 0.5, 0.99])
        res = least_squares(
            objective,
            initial_params,
            bounds=bounds,
            method='trf',
            max_nfev=5000,
            ftol=1e-8,
            gtol=1e-8,
            xtol=1e-8,
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

        result = CalibrationResult()
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
    def __init__(self):
        self.optimal_params = {}
        self.fitted_ivs = None
        self.market_ivs = None
        self.rmse = 0.0
        self.convergence_info = {}
        self.simulation_stats = {}

    def plot_fit_quality(self):
        fig, ax = plt.subplots()
        _unused = fig
        ax.plot(self.market_ivs.flatten(), label='Market IV')
        ax.plot(self.fitted_ivs.flatten(), label='Model IV')
        ax.legend()
        plt.show()

    def generate_report(self):
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
    initial_params = np.array([1.0, 0.1, -0.5])  # Initial guess for [eta, hurst, rho]

    logger.info("Pricing options...")
    option_prices = rbergomi_model.price_options(
        strikes=rbergomi_model.strikes,
        maturities=maturities,
        params=initial_params,
        n_paths=50000,
        n_steps=512,
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

    logger.info("Rough Bergomi model calculation completed.")


if __name__ == "__main__":
    main()
