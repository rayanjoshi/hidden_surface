"""Barrier option pricing using calibrated Rough Bergomi parameters.

This module implements European barrier option pricing (down-and-out, up-and-out,
down-and-in, up-and-in) using Monte Carlo simulation with the calibrated rBergomi model.

Features:
- Loads calibrated parameters from JSON
- Continuous barrier monitoring via fine time discretization
- Black-Scholes closed-form benchmarks for validation
- Comprehensive reporting and visualisation
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Literal, Optional
from scipy.stats import norm
from omegaconf import DictConfig
from numpy.random import default_rng
import hydra

from src.rbergomi.engine import RoughBergomiEngine
from src.rbergomi.convolution import precompute_convolution_matrix
from src.rbergomi.simulation import simulate_rbergomi_paths
from scripts.logging_config import get_logger, setup_logging


class BarrierOptionPricer:
    """
    Price barrier options using calibrated Rough Bergomi model.

    Attributes
    ----------
    engine : RoughBergomiEngine
        Core rBergomi engine for path simulation
    calibrated_params : Dict[str, float]
        Calibrated eta, hurst, rho from JSON
    logger : logging.Logger
        Logger instance
    """

    def __init__(self, cfg: DictConfig, calibration_json_path: str):
        """
        Initialize barrier pricer with calibrated parameters.

        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration
        calibration_json_path : str
            Path to JSON file containing calibration results
        """
        self.cfg = cfg
        self.logger = get_logger("BarrierOptionPricer")
        self.engine = RoughBergomiEngine(cfg)

        # Load calibrated parameters
        calibration_path = Path(calibration_json_path)
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

        with open(calibration_path, "r") as f:
            calibration_data = json.load(f)

        self.calibrated_params = calibration_data["Optimal Params"]
        self.logger.info(
            "Loaded calibrated parameters: eta=%.4f, H=%.4f, rho=%.4f",
            self.calibrated_params["eta"],
            self.calibrated_params["hurst"],
            self.calibrated_params["rho"],
        )

    def price_barrier_option(
        self,
        strike: float,
        barrier: float,
        maturity: float,
        barrier_type: Literal["down-and-out", "up-and-out", "down-and-in", "up-and-in"],
        option_type: Literal["call", "put"] = "put",
        n_paths: int = 100000,
        n_steps: int = 4096,
        rebate: float = 0.0,
    ) -> Dict[str, float]:
        """
        Price a barrier option using Monte Carlo simulation.

        Parameters
        ----------
        strike : float
            Strike price (normalized, typically relative to S0=1)
        barrier : float
            Barrier level (normalized)
        maturity : float
            Time to maturity in years
        barrier_type : str
            Type of barrier: 'down-and-out', 'up-and-out', 'down-and-in', 'up-and-in'
        option_type : str
            'call' or 'put'
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps (higher = better barrier monitoring)
        rebate : float
            Rebate paid if barrier is breached (default 0)

        Returns
        -------
        Dict[str, float]
            Dictionary containing price, standard error, and breach probability
        """
        params = (
            self.calibrated_params["eta"],
            self.calibrated_params["hurst"],
            self.calibrated_params["rho"],
        )

        # Simulate full paths (not just terminal values)
        self.logger.info(
            "Simulating %d paths with %d steps for barrier monitoring...",
            n_paths,
            n_steps,
        )

        eta, hurst, rho = params
        alpha = hurst - 0.5

        K = precompute_convolution_matrix(alpha, n_steps)

        # Time grid and forward variance
        t = np.linspace(0, maturity, n_steps + 1)
        xi = self.engine.forward_variance_func(t)
        short_rates = np.asarray(self.engine.get_short_rate(t), dtype=float)

        # Generate random numbers with antithetic variates
        seed = self.cfg.seed
        rng = default_rng(seed)
        n_base = n_paths // 2

        g1_base = rng.standard_normal((n_base, n_steps))
        all_g1 = np.vstack([g1_base, -g1_base])[:n_paths, :]

        z_base = rng.standard_normal((n_base, n_steps))
        all_z = np.vstack([z_base, -z_base])[:n_paths, :]

        # Simulate paths
        s_paths, variance_paths = simulate_rbergomi_paths(
            n_paths,
            n_steps,
            maturity,
            eta,
            hurst,
            rho,
            xi,
            all_g1,
            all_z,
            short_rates,
            K,
        )

        # Check barrier breaches
        breach_indicator = self._check_barrier_breach(s_paths, barrier, barrier_type)

        # Compute terminal payoffs
        s_terminal = s_paths[:, -1]

        if option_type == "call":
            intrinsic_payoff = np.maximum(s_terminal - strike, 0)
        else:  # put
            intrinsic_payoff = np.maximum(strike - s_terminal, 0)

        # Apply barrier logic
        if barrier_type in ["down-and-out", "up-and-out"]:
            # Knock-out: payoff only if barrier NOT breached
            payoff = np.where(breach_indicator, rebate, intrinsic_payoff)
        else:  # knock-in
            # Knock-in: payoff only if barrier IS breached
            payoff = np.where(breach_indicator, intrinsic_payoff, rebate)

        # Discount to present value
        r = float(self.engine.get_yield(maturity))
        discount = np.exp(-r * maturity)

        price = np.mean(payoff) * discount
        std_err = np.std(payoff) * discount / np.sqrt(n_paths)
        breach_probability = np.mean(breach_indicator)

        result = {
            "price": float(price),
            "std_error": float(std_err),
            "breach_probability": float(breach_probability),
            "discount_factor": float(discount),
        }

        self.logger.info(
            "Barrier option price: %.6f ± %.6f (breach probability: %.2f%%)",
            price,
            std_err,
            breach_probability * 100,
        )

        return result

    def _check_barrier_breach(
        self, paths: np.ndarray, barrier: float, barrier_type: str
    ) -> np.ndarray:
        """
        Check if barrier was breached for each path.

        Parameters
        ----------
        paths : np.ndarray
            Stock price paths (n_paths, n_steps+1)
        barrier : float
            Barrier level
        barrier_type : str
            Type of barrier

        Returns
        -------
        np.ndarray
            Boolean array indicating breach for each path
        """
        if barrier_type in ["down-and-out", "down-and-in"]:
            # Check if any point goes below barrier
            breached = np.any(paths <= barrier, axis=1)
        else:  # up-and-out, up-and-in
            # Check if any point goes above barrier
            breached = np.any(paths >= barrier, axis=1)

        return breached

    def black_scholes_barrier(
        self,
        s: float,
        k: float,
        barrier: float,
        t: float,
        r: float,
        sigma: float,
        barrier_type: Literal["down-and-out", "up-and-out"],
        option_type: Literal["call", "put"] = "put",
        rebate: float = 0.0,
    ) -> float:
        """
        Closed-form Black-Scholes barrier option price.

        Uses Merton (1973) / Reiner-Rubinstein (1991) formulas.

        Parameters
        ----------
        s : float
            Current spot price
        k : float
            Strike price
        barrier : float
            Barrier level
        t : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        barrier_type : str
            'down-and-out' or 'up-and-out'
        option_type : str
            'call' or 'put'
        rebate : float
            Rebate if barrier breached

        Returns
        -------
        float
            Barrier option price
        """
        if sigma <= 0 or t <= 0:
            return 0.0

        # Helper variables
        sqrt_t = np.sqrt(t)
        mu = (r - 0.5 * sigma**2) / (sigma**2)
        lambda_param = np.sqrt(mu**2 + 2 * r / (sigma**2))

        # Standard d1, d2
        d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        # Barrier-adjusted terms
        h = barrier
        y1 = (np.log(h**2 / (s * k)) / (sigma * sqrt_t)) + lambda_param * sigma * sqrt_t
        y2 = (np.log(h**2 / (s * k)) / (sigma * sqrt_t)) - lambda_param * sigma * sqrt_t

        if barrier_type == "down-and-out":
            if option_type == "put":
                if barrier >= k:
                    # Barrier above strike - formula becomes complex
                    # Use numerical approximation or return 0
                    return 0.0
                else:
                    # Standard down-and-out put (barrier < strike)
                    vanilla_put = k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)

                    # Adjustment term
                    adjustment = (h / s) ** (2 * lambda_param) * (
                        k * np.exp(-r * t) * norm.cdf(-y2) - h**2 / s * norm.cdf(-y1)
                    )

                    price = vanilla_put - adjustment
                    return max(price, 0.0)
            else:  # call
                # Down-and-out call formulas
                if barrier >= k:
                    return 0.0
                else:
                    vanilla_call = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
                    adjustment = (h / s) ** (2 * lambda_param) * (
                        h**2 / s * norm.cdf(y1) - k * np.exp(-r * t) * norm.cdf(y2)
                    )
                    price = vanilla_call - adjustment
                    return max(price, 0.0)

        elif barrier_type == "up-and-out":
            if option_type == "call":
                if barrier <= k:
                    return 0.0
                else:
                    vanilla_call = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
                    adjustment = (h / s) ** (2 * lambda_param) * (
                        h**2 / s * norm.cdf(y1) - k * np.exp(-r * t) * norm.cdf(y2)
                    )
                    price = vanilla_call - adjustment
                    return max(price, 0.0)
            else:  # put
                if barrier <= k:
                    return 0.0
                else:
                    vanilla_put = k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
                    adjustment = (h / s) ** (2 * lambda_param) * (
                        k * np.exp(-r * t) * norm.cdf(-y2) - h**2 / s * norm.cdf(-y1)
                    )
                    price = vanilla_put - adjustment
                    return max(price, 0.0)

        return 0.0

    def compare_to_black_scholes(
        self,
        strike: float,
        maturity: float,
        barrier_levels: np.ndarray,
        barrier_type: Literal["down-and-out", "up-and-out"],
        option_type: Literal["call", "put"] = "put",
        n_paths: int = 100000,
        n_steps: int = 4096,
    ) -> pd.DataFrame:
        """
        Compare rBergomi barrier prices to Black-Scholes across multiple barriers.

        Parameters
        ----------
        strike : float
            Strike price
        maturity : float
            Time to maturity
        barrier_levels : np.ndarray
            Array of barrier levels to test
        barrier_type : str
            'down-and-out' or 'up-and-out'
        option_type : str
            'call' or 'put'
        n_paths : int
            Number of MC paths
        n_steps : int
            Number of time steps

        Returns
        -------
        pd.DataFrame
            Comparison table with columns: barrier, rbergomi_price, bs_price, difference
        """
        results = []

        # Get ATM volatility for Black-Scholes benchmark
        atm_iv = self._get_atm_volatility(maturity)
        r = float(self.engine.get_yield(maturity))

        self.logger.info(
            "Comparing barrier prices for %d barrier levels...", len(barrier_levels)
        )

        for barrier in barrier_levels:
            # rBergomi price
            rb_result = self.price_barrier_option(
                strike=strike,
                barrier=barrier,
                maturity=maturity,
                barrier_type=barrier_type,
                option_type=option_type,
                n_paths=n_paths,
                n_steps=n_steps,
            )

            # Black-Scholes price
            bs_price = self.black_scholes_barrier(
                s=1.0,
                k=strike,
                barrier=barrier,
                t=maturity,
                r=r,
                sigma=atm_iv,
                barrier_type=barrier_type,
                option_type=option_type,
            )

            results.append(
                {
                    "barrier": barrier,
                    "rbergomi_price": rb_result["price"],
                    "rbergomi_stderr": rb_result["std_error"],
                    "bs_price": bs_price,
                    "difference": rb_result["price"] - bs_price,
                    "breach_probability": rb_result["breach_probability"],
                }
            )

            self.logger.info(
                "Barrier=%.3f: rBergomi=%.6f, BS=%.6f, diff=%.6f",
                barrier,
                rb_result["price"],
                bs_price,
                rb_result["price"] - bs_price,
            )

        return pd.DataFrame(results)

    def _get_atm_volatility(self, maturity: float) -> float:
        """
        Get ATM implied volatility for a given maturity.

        Parameters
        ----------
        maturity : float
            Time to maturity

        Returns
        -------
        float
            ATM implied volatility
        """
        # Find closest maturity in IV surface
        mat_idx = np.argmin(np.abs(self.engine.maturities - maturity))

        # Find ATM strike (closest to 1.0 for normalized grid)
        strike_idx = np.argmin(np.abs(self.engine.strikes - 1.0))

        atm_iv = self.engine.iv_surface[mat_idx, strike_idx]

        if np.isnan(atm_iv) or atm_iv <= 0:
            self.logger.warning("Invalid ATM IV, using 0.2 as fallback")
            return 0.2

        return float(atm_iv)

    def plot_barrier_comparison(
        self, comparison_df: pd.DataFrame, save_path: Optional[Path] = None
    ) -> None:
        """
        Plot comparison between rBergomi and Black-Scholes barrier prices.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            DataFrame from compare_to_black_scholes
        save_path : Path, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Prices
        ax1.plot(
            comparison_df["barrier"],
            comparison_df["rbergomi_price"],
            "o-",
            label="rBergomi",
            linewidth=2,
        )
        ax1.fill_between(
            comparison_df["barrier"],
            comparison_df["rbergomi_price"] - 2 * comparison_df["rbergomi_stderr"],
            comparison_df["rbergomi_price"] + 2 * comparison_df["rbergomi_stderr"],
            alpha=0.3,
            label="95% CI",
        )
        ax1.plot(
            comparison_df["barrier"],
            comparison_df["bs_price"],
            "s--",
            label="Black-Scholes",
            linewidth=2,
        )
        ax1.set_xlabel("Barrier Level", fontsize=12)
        ax1.set_ylabel("Option Price", fontsize=12)
        ax1.set_title("Barrier Option Prices: rBergomi vs Black-Scholes", fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Difference and breach probability
        ax2_twin = ax2.twinx()

        line1 = ax2.plot(
            comparison_df["barrier"],
            comparison_df["difference"],
            "o-",
            color="darkred",
            label="Price Difference",
            linewidth=2,
        )
        ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Barrier Level", fontsize=12)
        ax2.set_ylabel("Price Difference (rBergomi - BS)", fontsize=12, color="darkred")
        ax2.tick_params(axis="y", labelcolor="darkred")

        line2 = ax2_twin.plot(
            comparison_df["barrier"],
            comparison_df["breach_probability"] * 100,
            "s--",
            color="darkblue",
            label="Breach Probability",
            linewidth=2,
        )
        ax2_twin.set_ylabel("Breach Probability (%)", fontsize=12, color="darkblue")
        ax2_twin.tick_params(axis="y", labelcolor="darkblue")

        ax2.set_title("Price Difference and Breach Probability", fontsize=13)

        # Combine legends
        lines = line1 + line2
        labels = [L.get_label() for L in lines]
        ax2.legend(lines, labels, loc="best", fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
            self.logger.info("Plot saved to %s", save_path)


@hydra.main(
    version_base=None, config_path="../configs", config_name="rbergomi_model"
)
def main(cfg: DictConfig):
    """
    Main entry point for barrier option pricing using the calibrated Rough Bergomi model.

    Loads calibrated parameters, prices a selection of barrier options,
    compares results against Black-Scholes benchmarks (where closed-form exists),
    and generates comprehensive visualisation and reporting.
    """
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")

    logger.info("=" * 70)
    logger.info("BARRIER OPTION PRICING - ROUGH BERGOMI MODEL")
    logger.info("=" * 70)

    # Paths
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent

    # Load calibration results
    calibration_file = Path(
        repo_root / cfg.rbergomi.save_path / "calibration_report.json"
    )
    if not calibration_file.exists():
        logger.error("Calibration file not found: %s", calibration_file)
        logger.error(
            "Please run rbergomi_model.py first to generate calibrated parameters"
        )
        return

    # Initialize pricer
    pricer = BarrierOptionPricer(cfg, str(calibration_file))

    # Define option specifications to price
    maturity = cfg.barrier_option.maturity  # e.g. 1.0 year
    strike = cfg.barrier_option.strike  # normalised strike (ATM = 1.0)
    option_type = cfg.barrier_option.option_type  # 'put' or 'call'

    # Example barrier types and levels (can be extended in config)
    barrier_configs = cfg.barrier_option.barriers
    # Expected format in config:
    # barriers:
    #   down_and_out: [0.70, 0.80, 0.90, 0.95]
    #   up_and_out:   [1.05, 1.10, 1.20, 1.30]
    #   down_and_in:  [0.80, 0.90]
    #   up_and_in:    [1.10, 1.20]

    all_results = []

    # Price individual barrier options
    logger.info("\nPricing individual barrier options...")
    for barrier_type, barriers in barrier_configs.items():
        if not barriers:
            continue
        logger.info(
            f"\n--- {barrier_type.upper().replace('_', '-')} {option_type.upper()} ---"
        )
        for barrier in barriers:
            result = pricer.price_barrier_option(
                strike=strike,
                barrier=barrier,
                maturity=maturity,
                barrier_type=barrier_type,
                option_type=option_type,
                n_paths=cfg.barrier_option.n_paths,
                n_steps=cfg.barrier_option.n_steps,
                rebate=cfg.barrier_option.rebate,
            )
            result.update(
                {
                    "barrier_type": barrier_type.replace("_", "-"),
                    "barrier": barrier,
                    "maturity": maturity,
                    "strike": strike,
                    "option_type": option_type,
                }
            )
            all_results.append(result)

            logger.info(
                "Barrier %.3f: Price = %.6f ± %.6f  (breach probability = %.2f%%)",
                barrier,
                result["price"],
                result["std_error"],
                result["breach_probability"] * 100,
            )

    # Convert individual results to DataFrame
    individual_df = pd.DataFrame(all_results)

    # Comparison with Black-Scholes for knock-out barriers (closed-form available)
    comparison_results = []
    for barrier_type in ["down-and-out", "up-and-out"]:
        barriers = np.array(barrier_configs.get(barrier_type.replace("-", "_"), []))
        if len(barriers) == 0:
            continue

        logger.info(f"\nComparing {barrier_type.upper()} to Black-Scholes benchmark...")
        comp_df = pricer.compare_to_black_scholes(
            strike=strike,
            maturity=maturity,
            barrier_levels=barriers,
            barrier_type=barrier_type,
            option_type=option_type,
            n_paths=cfg.barrier_option.n_paths,
            n_steps=cfg.barrier_option.n_steps,
        )
        comparison_results.append(comp_df)

    comparison_df = (
        pd.concat(comparison_results, ignore_index=True) if comparison_results else None
    )

    # Save results
    save_path = Path(repo_root / cfg.rbergomi.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    individual_file = save_path / "barrier_option_results.csv"
    individual_df.to_csv(individual_file, index=False)
    logger.info("Individual barrier option results saved to %s", individual_file)

    if comparison_df is not None:
        comparison_file = save_path / "barrier_option_bs_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        logger.info("Black-Scholes comparison saved to %s", comparison_file)

    # Plot comparison (only for knock-out barriers)
    if comparison_df is not None and not comparison_df.empty:
        logger.info("\nGenerating comparison plot...")
        plot_file = save_path / "barrier_option_comparison.svg"
        pricer.plot_barrier_comparison(comparison_df, save_path=plot_file)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Maturity: %.2f years", maturity)
    logger.info("Strike: %.3f (normalised)", strike)
    logger.info("Option type: %s", option_type.upper())
    logger.info("Calibrated parameters:")
    logger.info("  η (vol-of-vol): %.4f", pricer.calibrated_params["eta"])
    logger.info("  H (Hurst):      %.4f", pricer.calibrated_params["hurst"])
    logger.info("  ρ (correlation): %.4f", pricer.calibrated_params["rho"])
    logger.info("\nPricing completed. Results saved in %s", save_path)


if __name__ == "__main__":
    main()
