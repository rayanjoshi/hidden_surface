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

from pathlib import Path
import json
import numpy as np
import hydra
from omegaconf import DictConfig

from rbergomi.engine import RoughBergomiEngine
from rbergomi.black_scholes import black_scholes_call
from rbergomi.black_scholes import black_scholes_put
from scripts.logging_config import get_logger, setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="rbergomi_model")
def main(cfg: DictConfig):
    """
    Main entry point for rbergomi model calculation.

    Initialises logging, creates a RoughBergomiEngine instance, and
    executes the full calculation pipeline.

    Args:
        cfg: configuration object.
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
