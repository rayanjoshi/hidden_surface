"""
Forward variance calculation module using the Carr-Madan method.

This module provides functionality to compute forward variance curves and variance
swap rates from market data. It uses configuration from Hydra, processes data with pandas, 
and applies smoothing with SciPy. Results are saved as CSV files and visualised with Matplotlib.
"""
from typing import Optional
from pathlib import Path
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import hydra
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end

class ForwardVarianceCalculator:
    """
    Calculates forward variance and variance swap rates from market data.

    This class implements the Carr-Madan method for computing forward variance
    curves and variance swap rates. It loads market data, processes it, applies
    smoothing, and saves results.

    Attributes:
        cfg (DictConfig): Configuration object containing parameters.
        logger (logging.Logger): Logger for tracking operations.
        input_path (Path): Path to input market data file.
        df (pd.DataFrame): Loaded market data.
        minimum_points (int): Minimum number of data points required.
        sigma (float): Smoothing parameter for Gaussian filter.
        forward_variance_curve (callable): Interpolated forward variance curve.
        theta_curve (callable): Interpolated variance swap rate curve.
        output_path (Path): Path for saving output files.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.logger = get_logger("ForwardVarianceCalculator")
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        input_path = self.cfg.select_date.output_path
        self.input_path = Path(repo_root / input_path).resolve()
        self.df = None
        self.minimum_points = self.cfg.forward_variance.minimum_points
        self.sigma = self.cfg.forward_variance.sigma
        self.forward_variance_curve = None
        self.theta_curve = None
        self.output_path = Path(repo_root / self.cfg.forward_variance.output_path).resolve()

    def load_data(self):
        """
        Load market data from the input path.
        """
        log_function_start("load_data")
        self.df = pd.read_csv(self.input_path)
        self.logger.info("Loaded market data from %s", self.input_path)
        log_function_end("load_data")

    def carr_madan_forward_variance(self):
        """
        Computes forward variance curve using the Carr-Madan method.

        Uses model-free variance swap pricing (Carr-Madan) with strict filtering,
        light smoothing, and monotonic spline on cumulative variance. W(t) = t * theta(t).
        """
        log_function_start("carr_madan_forward_variance")
        self.logger.info("Calculating forward variance using Carr-Madan method")

        maturity_groups = self.df.groupby("T_years")

        var_swaps = []
        maturities = []

        for maturity, group in maturity_groups:
            n_calls = len(group[group["cp_flag"] == "C"])
            n_puts = len(group[group["cp_flag"] == "P"])
            if n_calls + n_puts < 20 or n_calls < 5 or n_puts < 5:
                self.logger.warning(
                    "Insufficient calls (%d) or puts (%d) for maturity %f, skipping...",
                    n_calls, n_puts, maturity
                )
                continue

            forward = group["underlying_close"].iloc[0] * np.exp(
                group["risk_free_rate"].iloc[0] * maturity
            )
            theta = self._variance_swap_rate(group, forward, maturity)
            if 0.01 < theta < 20.0:
                maturities.append(maturity)
                var_swaps.append(theta)
                
        if len(maturities) < 5:
            self.logger.error("No valid maturities found; using constant fallback.")
            self.forward_variance_curve = lambda t: 0.225  # Default to 20% vol squared
            return

        T = np.array(maturities)
        theta = np.array(var_swaps)
        
        if len(theta) > 5:
            theta = gaussian_filter1d(theta, sigma=self.sigma)
        
        W = T * theta
        T_ext = np.concatenate(([1e-8], T, [T.max() + 10.0]))  # extend far
        W_ext = np.concatenate(([0.0], W, [W[-1] + 10.0 * theta[-1]]))  # linear extrapolation

        # Monotonic PCHIP spline (preserves shape, never decreases)
        spl_W = interpolate.PchipInterpolator(T_ext, W_ext)

        # Dense grid
        t_dense = np.logspace(-3, np.log10(T.max() + 20), 5000)

        # Analytical derivative of PCHIP is stable and always positive
        xi_dense = spl_W.derivative()(t_dense)

        # Only light cleaning — never floor to 0.04!
        xi_dense = np.maximum(xi_dense, 0.001)  # only prevent crazy negatives
        xi_dense = np.minimum(xi_dense, 10.0)   # cap insane values

        # Final interpolator (linear is more stable than spline derivative)
        long_term_level = xi_dense[-1000:].mean()       # typically ~0.21–0.23 for TSLA
        self.forward_variance_curve = interpolate.interp1d(
            t_dense,
            xi_dense,
            kind='linear',
            bounds_error=False,
            fill_value=long_term_level,
        )
        
        self.theta_curve = interpolate.interp1d(
            T,
            theta,
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )

        self.logger.info(f"Forward variance ξ(0+) ≈ {xi_dense[0]:.4f} "
                        f"({np.sqrt(xi_dense[0])*100:.1f}% instantaneous vol)")
        self.logger.info(f"Forward variance ξ(1Y) ≈ {self.forward_variance_curve(1.0):.4f}")

    def _variance_swap_rate(self, df, forward_price, maturity):

        calls = df[df["cp_flag"] == "C"].copy()
        puts = df[df["cp_flag"] == "P"].copy()

        if len(calls) == 0 and len(puts) == 0:
            return 0.0

        calls = calls.sort_values(by="strike_price")
        puts = puts.sort_values(by="strike_price")

        otm_calls = calls[calls["strike_price"] > forward_price]
        otm_puts = puts[puts["strike_price"] < forward_price]
        total_otm = len(otm_calls) + len(otm_puts)

        if total_otm < self.minimum_points:
            self.logger.warning("Too few OTM options (%d) for maturity=%f; using ATM approximation",
                                total_otm, maturity)
            return self._atm_approx(df, maturity)

        var_contribution = 0.0

        # OTM calls
        if not otm_calls.empty:
            strikes = otm_calls["strike_price"].to_numpy()
            prices = otm_calls["mid"].to_numpy()
            idx = np.argsort(strikes)
            strikes = strikes[idx]
            prices = prices[idx]
            integrand = prices / strikes**2
            if len(strikes) > 1:
                var_contribution += np.trapezoid(integrand, strikes)
            else:
                var_contribution += integrand[0] * (strikes[0] * 0.01)  # Approx dK as 1% of K

        # OTM puts
        if not otm_puts.empty:
            strikes = otm_puts["strike_price"].to_numpy()
            prices = otm_puts["mid"].to_numpy()
            idx = np.argsort(strikes)
            strikes = strikes[idx]
            prices = prices[idx]
            integrand = prices / strikes**2
            if len(strikes) > 1:
                var_contribution += np.trapezoid(integrand, strikes)
            else:
                var_contribution += integrand[0] * (strikes[0] * 0.01)  # Approx

        if var_contribution <= 0:
            self.logger.warning(
                "Non-positive var_contribution; using ATM approximation for maturity=%f",
                maturity,
            )
            return self._atm_approx(df, maturity)

        r = df["risk_free_rate"].iloc[0]
        theta = (2 * np.exp(r * maturity) * var_contribution) / maturity

        return max(theta, 0.0)

    def _atm_approx(self, df, maturity):
        _unused = maturity
        # Approximate theta ≈ ATM IV²
        atm_idx = df['log_moneyness'].abs().idxmin()
        iv_atm = df.loc[atm_idx, 'implied_volatility']
        return iv_atm ** 2

    def save_data(self):
        """
        Saves forward variance and variance swap rate data to a CSV file.

        Creates a DataFrame with maturity, forward variance, and variance swap rate
        values, and saves it to the configured output path.
        """
        log_function_start("save_data")
        if self.forward_variance_curve is None:
            self.logger.error("Cannot save: forward variance curve not computed.")
            log_function_end("save_data")
            return

        max_maturity = max(self.df["T_years"])
        t_plot = np.linspace(0.01, max_maturity, 100)
        forward_variance = [self.forward_variance_curve(t) for t in t_plot]

        theta_vol = [np.nan] * len(t_plot)
        if self.theta_curve is not None:
            theta_var = np.array([self.theta_curve(t) for t in t_plot])
            theta_vol = np.sqrt(np.maximum(theta_var, 0.0))

        output_df = pd.DataFrame({
            "maturity_years": t_plot,
            "forward_variance": forward_variance,
            "variance_swap_rate": theta_vol
        })

        self.output_path.mkdir(parents=True, exist_ok=True)
        output_file = self.output_path / "forward_variance_data.csv"
        output_df.to_csv(output_file, index=False)
        self.logger.info("Saved forward variance and theta curves to %s", output_file)

        log_function_end("save_data")

    def plot_variance(self):
        """
        Plots forward variance and variance swap rate curves.

        Generates a dual-axis plot with forward variance and variance swap rates,
        saving the figure to the output path.
        """
        log_function_start("plot_variance")

        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        max_maturity = max(self.df["T_years"])
        t_plot = np.linspace(0.01, max_maturity, 100)
        forward_variance = [self.forward_variance_curve(t) for t in t_plot]
        
        ax1.plot(t_plot, forward_variance, label="Forward Variance ξ(t)", color="blue")
        ax1.set_xlabel("Maturity (Years)")
        ax1.set_ylabel("Forward Variance", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        ax1.grid()

        ax2 = ax1.twinx()
        theta_plot = [self.theta_curve(t) for t in t_plot[1:]]
        ax2.plot(t_plot[1:], theta_plot, label="Variance Swap θ(t)", color="green", linestyle="--")
        ax2.set_ylabel("Variance Swap Rate", color="green")
        ax2.tick_params(axis='y', labelcolor="green")

        fig.suptitle("Forward Variance and Variance Swap Curves")
        fig.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.output_path / "forward_variance.svg", format='svg')
        plt.savefig(self.output_path / "forward_variance.png", format='png')

        log_function_end("plot_variance")

@hydra.main(version_base=None, config_path="../configs", config_name="forward_variance")
def main(cfg: Optional[DictConfig] = None):
    """
    Main entry point for forward variance calculation.

    Initialises logging, creates a ForwardVarianceCalculator instance, and
    executes the full calculation pipeline.

    Args:
        cfg: Optional configuration object; defaults to None.
    """
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")
    logger.info("Starting forward variance calculation...")
    calculator = ForwardVarianceCalculator(cfg)
    calculator.load_data()
    calculator.carr_madan_forward_variance()
    calculator.save_data()
    calculator.plot_variance()

if __name__ == "__main__":
    main()
