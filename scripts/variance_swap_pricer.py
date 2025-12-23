"""Variance Swap Pricer for Rough Bergomi Model

This module prices variance swaps using Monte Carlo simulation with calibrated
rBergomi parameters and compares against the Carr-Madan theoretical variance swap rate.

Variance Swap Payoff:
    Payoff = Realised Variance - Strike

where Realised Variance = (1/T) ∫₀ᵀ Vₜ dt

The fair strike (variance swap rate) is E[Realised Variance], which under rBergomi
should match the Carr-Madan rate computed from the forward variance curve.

Usage:
    python variance_swap_pricer.py
"""

import json
from pathlib import Path
from typing import Tuple, Any, cast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import hydra
from omegaconf import DictConfig

from src.rbergomi.convolution import precompute_convolution_matrix
from src.rbergomi.simulation import simulate_rbergomi_paths
from scripts.logging_config import get_logger, setup_logging


class VarianceSwapPricer:
    """
    Price variance swaps under the calibrated Rough Bergomi model.

    Attributes:
        cfg: Hydra configuration
        calibrated_params: Dictionary containing eta, hurst, rho
        forward_variance_func: Interpolator for ξ(t)
        maturities: Array of variance swap maturities to price
    """

    def __init__(self, cfg: DictConfig, calibration_file: Path):
        """
        Initialize pricer with calibrated parameters.

        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration
        calibration_file : Path
            Path to calibration_report.json containing optimal parameters
        """
        self.cfg = cfg
        self.logger = get_logger("VarianceSwapPricer")

        # Load calibrated parameters
        self.logger.info("Loading calibrated parameters from %s", calibration_file)
        with open(calibration_file, "r") as f:
            calibration_report = json.load(f)

        self.calibrated_params = calibration_report["Optimal Params"]
        self.eta = self.calibrated_params["eta"]
        self.hurst = self.calibrated_params["hurst"]
        self.rho = self.calibrated_params["rho"]

        self.logger.info(
            "Loaded parameters: η=%.4f, H=%.4f, ρ=%.4f", self.eta, self.hurst, self.rho
        )

        # Load forward variance curve
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent.parent
        f_variance_path = Path(repo_root / cfg.f_variance.output_path).resolve()

        self.logger.info("Loading forward variance from %s", f_variance_path)
        self.df = pd.read_csv(f_variance_path)

        self.forward_variance_func = interp1d(
            self.df["maturity_years"],
            self.df["forward_variance"],
            bounds_error=False,
            fill_value=cast(
                Any,
                (
                    float(self.df["forward_variance"].iloc[0]),
                    float(self.df["forward_variance"].iloc[-1]),
                ),
            ),
        )

        # Maturities to price (monthly up to 2 years)
        self.maturities = np.array(
            [1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5, 2.0]
        )

    def compute_carr_madan_rate(self, maturity: float) -> float:
        """
        Compute theoretical variance swap rate using Carr-Madan formula.

        Under the rBergomi model (or any model with forward variance), the
        fair variance swap strike is:

            θ(T) = (1/T) ∫₀ᵀ ξ(t) dt

        where ξ(t) is the forward variance curve.

        Parameters
        ----------
        maturity : float
            Variance swap maturity in years

        Returns
        -------
        float
            Theoretical variance swap rate (annualised)
        """
        # Integrate forward variance using trapezoidal rule
        n_points = max(1000, int(maturity * 5000))
        t_grid = np.linspace(0, maturity, n_points)
        xi_grid = self.forward_variance_func(t_grid)

        # (1/T) ∫₀ᵀ ξ(t) dt
        integral = np.trapezoid(xi_grid, t_grid)
        return integral / maturity

    def price_variance_swap_mc(
        self,
        maturity: float,
        n_paths: int = 100000,
        n_steps: int = 2048,
        seed: int = 42,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Price variance swap via Monte Carlo simulation.

        The realised variance is computed as:
            RV = (1/T) ∫₀ᵀ Vₜ dt ≈ (1/T) Σᵢ Vₜᵢ Δt

        The fair strike (variance swap rate) is E[RV].

        Parameters
        ----------
        maturity : float
            Variance swap maturity in years
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps per path
        seed : int
            Random seed for reproducibility

        Returns
        -------
        variance_swap_rate : float
            Monte Carlo estimate of E[Realised Variance]
        std_error : float
            Standard error of the estimate
        realised_variances : np.ndarray
            Array of realised variances from all paths (for diagnostics)
        """
        self.logger.info(
            "Pricing variance swap: T=%.3f, n_paths=%d, n_steps=%d",
            maturity,
            n_paths,
            n_steps,
        )

        # Setup
        alpha = self.hurst - 0.5

        # Precompute convolution matrix
        K = precompute_convolution_matrix(alpha, n_steps)

        # Time grid and forward variance
        t = np.linspace(0, maturity, n_steps + 1)
        xi = self.forward_variance_func(t)

        # Short rates (set to zero for simplicity, variance swap doesn't need discounting)
        short_rates = np.zeros(n_steps + 1)

        # Generate random numbers with antithetic variates
        rng = np.random.default_rng(seed)
        n_base = n_paths // 2

        g1_base = rng.standard_normal((n_base, n_steps))
        all_g1 = np.vstack([g1_base, -g1_base])[:n_paths, :]

        z_base = rng.standard_normal((n_base, n_steps))
        all_z = np.vstack([z_base, -z_base])[:n_paths, :]

        # Simulate paths
        self.logger.info("Simulating rBergomi paths...")
        _, variance_paths = simulate_rbergomi_paths(
            n_paths=n_paths,
            n_steps=n_steps,
            maturity=maturity,
            eta=self.eta,
            hurst=self.hurst,
            rho=self.rho,
            f_variance=xi,
            all_g1=all_g1,
            all_z=all_z,
            short_rates=short_rates,
            K=K,
        )

        # Compute realised variance for each path
        # RV = (1/T) ∫₀ᵀ Vₜ dt ≈ (1/T) * sum(V[i] * dt)
        # Using trapezoidal rule: (dt/2) * (V[0] + 2*sum(V[1:-1]) + V[-1])
        realised_variances = np.zeros(n_paths)
        for i in range(n_paths):
            # Trapezoidal integration
            integral = np.trapezoid(variance_paths[i, :], t)
            realised_variances[i] = integral / maturity

        # Variance swap rate = E[RV]
        variance_swap_rate = float(np.mean(realised_variances))
        std_error = float(np.std(realised_variances) / np.sqrt(n_paths))

        self.logger.info(
            "MC variance swap rate: %.6f ± %.6f", variance_swap_rate, std_error
        )

        return (
            variance_swap_rate,
            std_error,
            np.asarray(realised_variances, dtype=float),
        )

    def price_all_maturities(
        self, n_paths: int = 100000, n_steps: int = 2048
    ) -> pd.DataFrame:
        """
        Price variance swaps across all maturities.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths per maturity
        n_steps : int
            Number of time steps

        Returns
        -------
        pd.DataFrame
            Results with columns: maturity, mc_rate, std_error, cm_rate, difference
        """
        results = []

        for maturity in self.maturities:
            self.logger.info("=" * 60)
            self.logger.info("Pricing T = %.3f years", maturity)

            # Monte Carlo pricing
            mc_rate, std_error, _ = self.price_variance_swap_mc(
                maturity=maturity, n_paths=n_paths, n_steps=n_steps, seed=self.cfg.seed
            )

            # Carr-Madan theoretical rate
            cm_rate = self.compute_carr_madan_rate(maturity)

            # Difference (should be small if calibration is good)
            diff = mc_rate - cm_rate
            diff_pct = 100 * diff / cm_rate

            self.logger.info("Carr-Madan rate: %.6f", cm_rate)
            self.logger.info("Difference: %.6f (%.2f%%)", diff, diff_pct)

            results.append(
                {
                    "maturity": maturity,
                    "mc_rate": mc_rate,
                    "std_error": std_error,
                    "cm_rate": cm_rate,
                    "difference": diff,
                    "difference_pct": diff_pct,
                }
            )

        return pd.DataFrame(results)

    def plot_results(self, results: pd.DataFrame, save_path: Path):
        """
        Create visualisation of variance swap pricing results.

        Parameters
        ----------
        results : pd.DataFrame
            Pricing results from price_all_maturities
        save_path : Path
            Directory to save plots
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        # Plot 1: MC vs Carr-Madan rates
        ax1 = axes[0]
        ax1.plot(
            results["maturity"],
            results["mc_rate"],
            "o-",
            label="Monte Carlo",
            linewidth=2,
            markersize=8,
        )
        ax1.plot(
            results["maturity"],
            results["cm_rate"],
            "s--",
            label="Carr-Madan (Theoretical)",
            linewidth=2,
            markersize=8,
        )
        ax1.fill_between(
            results["maturity"],
            results["mc_rate"] - 2 * results["std_error"],
            results["mc_rate"] + 2 * results["std_error"],
            alpha=0.3,
            label="95% CI",
        )
        ax1.set_xlabel("Maturity (years)", fontsize=12)
        ax1.set_ylabel("Variance Swap Rate (annualised)", fontsize=12)
        ax1.set_title(
            "Variance Swap Pricing: MC vs Theoretical", fontsize=14, fontweight="bold"
        )
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Pricing errors
        ax2 = axes[1]
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax2.bar(
            results["maturity"],
            results["difference_pct"],
            width=0.05,
            alpha=0.7,
            color="steelblue",
        )
        ax2.set_xlabel("Maturity (years)", fontsize=12)
        ax2.set_ylabel("Pricing Error (%)", fontsize=12)
        ax2.set_title(
            "MC vs Carr-Madan: Percentage Difference", fontsize=14, fontweight="bold"
        )
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Save plot
        save_path.mkdir(parents=True, exist_ok=True)
        plot_file = save_path / "variance_swap_pricing.svg"
        plt.savefig(plot_file, format="svg", bbox_inches="tight")
        self.logger.info("Plot saved to %s", plot_file)

        plt.close()


@hydra.main(
    version_base=None, config_path="../configs", config_name="rbergomi_model"
)
def main(cfg: DictConfig):
    """
    Main entry point for variance swap pricing.

    Loads calibrated parameters, prices variance swaps across maturities,
    and compares Monte Carlo results against Carr-Madan theoretical rates.
    """
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")

    logger.info("=" * 70)
    logger.info("VARIANCE SWAP PRICING - ROUGH BERGOMI MODEL")
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
    pricer = VarianceSwapPricer(cfg, calibration_file)

    # Price variance swaps
    logger.info("\nPricing variance swaps across maturities...")
    results = pricer.price_all_maturities(
        n_paths=cfg.variance_swap.n_paths, n_steps=cfg.variance_swap.n_steps
    )

    # Display results
    logger.info("\n" + "=" * 70)
    logger.info("VARIANCE SWAP PRICING RESULTS")
    logger.info("=" * 70)
    logger.info("\n%s\n", results.to_string(index=False))

    # Save results
    save_path = Path(repo_root / cfg.rbergomi.save_path)
    results_file = save_path / "variance_swap_results.csv"
    results.to_csv(results_file, index=False)
    logger.info("Results saved to %s", results_file)

    # Create visualisation
    logger.info("\nGenerating plots...")
    pricer.plot_results(results, save_path)

    # Summary statistics
    mean_error = results["difference_pct"].mean()
    max_error = results["difference_pct"].abs().max()
    rmse = np.sqrt((results["difference"] ** 2).mean())

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Mean pricing error: %.3f%%", mean_error)
    logger.info("Max absolute error: %.3f%%", max_error)
    logger.info("RMSE: %.6f", rmse)
    logger.info("\nCalibrated parameters:")
    logger.info("  η (vol-of-vol): %.4f", pricer.eta)
    logger.info("  H (Hurst):      %.4f", pricer.hurst)
    logger.info("  ρ (correlation): %.4f", pricer.rho)


if __name__ == "__main__":
    main()
