"""Container and reporting utilities for Rough Bergomi calibration results.

The :class:`CalibrationResult` class stores the outcome of a Rough Bergomi
calibration, provides diagnostic plotting, and generates structured reports
suitable for logging or saving to JSON.

Classes
-------
CalibrationResult
    Stores parameters, fitted surfaces, convergence info and offers plotting/reporting.
"""
from typing import Dict, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from scipy.optimize import OptimizeResult
from scripts.logging_config import get_logger


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
        repo_root = script_dir.parent.parent
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
        svg_path = self.save_path / "market_vs_model_iv.svg"
        png_path = self.save_path / "market_vs_model_iv.png"
        plt.savefig(png_path, format="png", dpi = 1200)
        plt.savefig(svg_path, format="svg")
        plt.close('all')
    
        np.save(self.save_path / "fitted_ivs.npy", self.fitted_ivs)
        np.save(self.save_path / "market_ivs.npy", self.market_ivs)
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
