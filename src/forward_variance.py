"""
Forward Variance Calculation Module
===================================

This module provides forward variance curve construction using the Carr-Madan
method with exponential spline parametrisation.

Mathematical Framework
----------------------
Forward variance curve ξ(t) is parameterized as:

    ξ(t) = exp(s(t))

where s(t) is a cubic spline with knots at market maturity quantiles.

The spline coefficients are fit by minimising:

    Σᵢ wᵢ [log(θ_model(Tᵢ)) - log(θ_market(Tᵢ))]² + λ * Roughness(s)

where:
- θ_model(T) = (1/T) ∫₀ᵀ exp(s(t)) dt (computed via numerical integration)
- θ_market(T) is the market variance swap rate from Carr-Madan
- λ is the smoothness penalty weight
- Roughness(s) = ∫(s''(t))² dt (penalises second derivative)
"""

from typing import Optional, Callable
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
import pandas as pd
import hydra
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.integrate import fixed_quad
import matplotlib.pyplot as plt

from scripts.logging_config import (
    get_logger,
    setup_logging,
    log_function_start,
    log_function_end,
)


@dataclass
class ForwardVarianceFitResult:
    """Result container for forward variance curve fitting."""

    success: bool
    rmse_log_space: float
    n_iterations: int
    coefficients: NDArray[np.float64]
    knots: NDArray[np.float64]
    message: str


class ForwardVarianceCurve:
    """
    Forward variance ξ(t) with guaranteed positivity via exponential transformation.

    Mathematical form: ξ(t) = exp(s(t))
    where s(t) is a cubic spline with knots at market maturity quantiles.

    Properties guaranteed:
    - ξ(t) > 0 for all t (exponential)
    - Smoothness controlled by spline order and regularisation
    - Proper integration: θ(T) = (1/T) ∫₀ᵀ ξ(t) dt

    Attributes
    ----------
    knots : np.ndarray
        Internal spline knot points (excluding boundary knots)
    lambda_reg : float
        Smoothness penalty weight λ for ∫(s''(t))² dt
    spline_tck : tuple or None
        Scipy spline representation (t, c, k) after fitting
    t_min : float
        Minimum time value in the fitted domain
    t_max : float
        Maximum time value in the fitted domain
    """

    def __init__(
        self,
        n_internal_knots: int = 8,
        regularisation_lambda: float = 0.01,
        extrapolation_method: str = "constant",
    ):
        """
        Initialize ForwardVarianceCurve.

        Parameters
        ----------
        n_internal_knots : int
            Number of internal spline knots (will be placed at quantiles of market data)
        regularisation_lambda : float
            Smoothness penalty weight λ for roughness term ∫(s''(t))² dt
            Recommended range: [0.001, 0.1]. Start with 0.01.
        extrapolation_method : str
            How to extrapolate beyond fitted domain: 'constant' or 'linear'
        """
        self.n_internal_knots = n_internal_knots
        self.lambda_reg = regularisation_lambda
        self.extrapolation_method = extrapolation_method

        # Will be set during fitting
        self.spline_tck: Optional[tuple] = None
        self.knots: Optional[NDArray[np.float64]] = None
        self.t_min: float = 0.0
        self.t_max: float = 10.0
        self._long_term_level: float = 0.04  # 20% vol squared default

        self.logger = get_logger("ForwardVarianceCurve")

    def fit(
        self,
        T_market: NDArray[np.float64],
        theta_market: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]] = None,
    ) -> ForwardVarianceFitResult:
        """
        Fit ξ(t) to market variance swap rates using exponential spline parameterization.

        Minimises:
            Σᵢ wᵢ [log(θ_model(Tᵢ)) - log(θ_market(Tᵢ))]² + λ * Roughness(s)

        where θ_model(T) = (1/T) ∫₀ᵀ exp(s(t)) dt is computed via numerical integration.

        Parameters
        ----------
        T_market : np.ndarray
            Market maturities where variance swaps are observed (must be positive)
        theta_market : np.ndarray
            Market variance swap rates θ(T) = W(T)/T (must be positive)
        weights : np.ndarray, optional
            Fitting weights (default: uniform)
            Recommended: based on number of options at each maturity

        Returns
        -------
        ForwardVarianceFitResult
            Fitting result with diagnostics
        """
        # Input validation
        T_market = np.asarray(T_market, dtype=np.float64)
        theta_market = np.asarray(theta_market, dtype=np.float64)

        if len(T_market) < 3:
            raise ValueError(f"Need at least 3 maturities, got {len(T_market)}")

        valid_mask = (T_market > 0) & (theta_market > 0) & np.isfinite(theta_market)
        if not np.all(valid_mask):
            self.logger.warning(
                "Filtering %d invalid data points (non-positive or NaN)",
                np.sum(~valid_mask),
            )
            T_market = T_market[valid_mask]
            theta_market = theta_market[valid_mask]

        if len(T_market) < 3:
            raise ValueError("Insufficient valid data points after filtering")

        # Sort by maturity
        sort_idx = np.argsort(T_market)
        T_market = T_market[sort_idx]
        theta_market = theta_market[sort_idx]

        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)[sort_idx]
        else:
            weights = np.ones_like(T_market)

        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)

        # Store domain bounds
        self.t_min = 0.0  # Always start from 0
        self.t_max = T_market[-1] * 1.5  # Extend slightly beyond last maturity

        # Create knots at quantiles of market maturities
        # Add boundary knots at 0 and t_max
        n_knots = min(self.n_internal_knots, len(T_market) - 1)
        if n_knots < 2:
            n_knots = 2

        quantiles = np.linspace(0, 100, n_knots + 2)
        internal_knots = np.percentile(T_market, quantiles[1:-1])

        # Ensure knots are strictly increasing and within bounds
        internal_knots = np.unique(internal_knots)
        internal_knots = internal_knots[
            (internal_knots > 1e-6) & (internal_knots < self.t_max)
        ]

        if len(internal_knots) < 2:
            # Fallback to uniform spacing
            internal_knots = np.linspace(T_market[0], T_market[-1], n_knots)

        self.knots = internal_knots

        # Full knot vector for cubic spline (k=3)
        # Need boundary knots with multiplicity 4 for cubic splines
        k = 3  # cubic spline degree
        full_knots = np.concatenate(
            [[self.t_min] * (k + 1), internal_knots, [self.t_max] * (k + 1)]
        )

        n_coefficients = len(full_knots) - k - 1

        # Initial guess: constant log-variance at mean of log(theta)
        log_theta_mean = np.mean(np.log(theta_market))
        initial_coefficients = np.full(n_coefficients, log_theta_mean)

        # Pre-compute integration points for efficiency
        # Use Gauss-Legendre quadrature on each maturity
        n_quad = 15  # Number of quadrature points

        def compute_theta_model(coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
            """Compute θ_model(T) = (1/T) ∫₀ᵀ exp(s(t)) dt for all maturities."""
            # Create spline from coefficients
            spline = BSpline(full_knots, coefficients, k, extrapolate=True)

            theta_model = np.zeros_like(T_market)
            for i, T in enumerate(T_market):
                if T <= 1e-10:
                    theta_model[i] = np.exp(spline(1e-10))
                    continue

                # Gauss-Legendre quadrature for ∫₀ᵀ exp(s(t)) dt
                # Transform [0, T] to [-1, 1]: t = (T/2)(x + 1)
                integral, _ = fixed_quad(
                    lambda x: np.exp(spline((T / 2) * (x + 1))), -1, 1, n=n_quad
                )
                integral *= T / 2  # Jacobian
                theta_model[i] = integral / T

            return theta_model

        def compute_roughness(coefficients: NDArray[np.float64]) -> float:
            """
            Compute roughness penalty ∫(s''(t))² dt.

            For cubic B-splines, the second derivative is piecewise linear and
            may have discontinuities at knot locations. We use dense trapezoidal
            integration which is more robust than adaptive quadrature for this case.
            """
            spline = BSpline(full_knots, coefficients, k, extrapolate=True)
            second_derivative = spline.derivative(2)

            # Dense trapezoidal integration - robust for piecewise linear functions
            n_points = 500
            t_dense = np.linspace(self.t_min + 1e-8, self.t_max - 1e-8, n_points)
            s_pp = second_derivative(t_dense)
            roughness = np.trapezoid(s_pp**2, t_dense)

            return roughness

        def objective(coefficients: NDArray[np.float64]) -> float:
            """
            Compute weighted log-space error + smoothness penalty.

            Loss = Σᵢ wᵢ [log(θ_model(Tᵢ)) - log(θ_market(Tᵢ))]² + λ * Roughness(s)
            """
            theta_model = compute_theta_model(coefficients)

            # Check for invalid values
            if np.any(theta_model <= 0) or np.any(np.isnan(theta_model)):
                return 1e10  # Large penalty for invalid configurations

            # Log-space errors (weighted)
            log_errors = np.log(theta_model) - np.log(theta_market)
            data_term = np.sum(weights * log_errors**2)

            # Roughness penalty
            roughness = compute_roughness(coefficients)
            penalty_term = self.lambda_reg * roughness

            return data_term + penalty_term

        def objective_gradient(coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
            """
            Compute gradient of objective via finite differences.

            Note: Analytical gradient is complex due to integration;
            finite differences are stable and sufficiently accurate.
            """
            eps = 1e-6
            grad = np.zeros_like(coefficients)
            f0 = objective(coefficients)

            for i in range(len(coefficients)):
                coefficients_plus = coefficients.copy()
                coefficients_plus[i] += eps
                grad[i] = (objective(coefficients_plus) - f0) / eps

            return grad

        # Optimisation using L-BFGS-B (works well for smooth objectives)
        self.logger.info(
            "Fitting forward variance curve: %d coefficients, λ=%.4f",
            n_coefficients,
            self.lambda_reg,
        )

        result = minimize(
            objective,
            initial_coefficients,
            method="L-BFGS-B",
            jac=objective_gradient,
            options={"maxiter": 2000, "ftol": 1e-8, "gtol": 1e-6},
        )

        # Store fitted spline
        self.spline_tck = (full_knots, result.x, k)

        # Compute long-term level for extrapolation
        spline = BSpline(full_knots, result.x, k, extrapolate=True)
        self._long_term_level = float(np.exp(spline(self.t_max * 0.9)))

        # Compute fit quality
        theta_fitted = compute_theta_model(result.x)
        rmse_log = float(
            np.sqrt(np.mean((np.log(theta_fitted) - np.log(theta_market)) ** 2))
        )

        self.logger.info(
            "Forward variance fit complete: RMSE (log-space) = %.6f, iterations = %d",
            rmse_log,
            result.nit,
        )

        return ForwardVarianceFitResult(
            success=result.success,
            rmse_log_space=rmse_log,
            n_iterations=result.nit,
            coefficients=result.x,
            knots=internal_knots,
            message=result.message,
        )

    def evaluate(self, t: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """
        Evaluate ξ(t) = exp(s(t)) at given times.

        For t outside the fitted domain:
        - t < t_min: uses value at t_min (constant extrapolation)
        - t > t_max: uses constant or linear extrapolation based on settings

        Parameters
        ----------
        t : np.ndarray or float
            Time(s) at which to evaluate forward variance

        Returns
        -------
        np.ndarray or float
            Forward variance values ξ(t) (guaranteed > 0)
        """
        if self.spline_tck is None:
            raise ValueError("Curve not fitted. Call fit() first.")

        scalar_input = np.isscalar(t)
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))

        knots, coefficients, k = self.spline_tck
        spline = BSpline(knots, coefficients, k, extrapolate=False)

        result = np.zeros_like(t)

        # In-domain evaluation
        in_domain = (t >= self.t_min) & (t <= self.t_max)
        if np.any(in_domain):
            s_t = spline(t[in_domain])
            result[in_domain] = np.exp(s_t)

        # Handle t < t_min (short end extrapolation)
        below_min = t < self.t_min
        if np.any(below_min):
            s_at_min = float(spline(max(self.t_min, 1e-8)))
            result[below_min] = np.exp(s_at_min)

        # Handle t > t_max (long end extrapolation)
        above_max = t > self.t_max
        if np.any(above_max):
            if self.extrapolation_method == "constant":
                result[above_max] = self._long_term_level
            else:  # linear extrapolation in xi-space
                # Get slope at t_max
                t_near = self.t_max - 0.1
                xi_near = np.exp(spline(t_near))
                slope = (self._long_term_level - xi_near) / 0.1
                result[above_max] = self._long_term_level + slope * (
                    t[above_max] - self.t_max
                )
                result[above_max] = np.maximum(
                    result[above_max], 0.01
                )  # Safety floor for extrapolation only

        if scalar_input:
            return float(result[0])
        return result

    def get_variance_swap_rate(
        self, T: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """
        Compute θ(T) = (1/T) ∫₀ᵀ ξ(t) dt via numerical integration.

        This is the variance swap rate implied by the forward variance curve.

        Parameters
        ----------
        T : float or np.ndarray
            Maturity/maturities for variance swap rate

        Returns
        -------
        float or np.ndarray
            Variance swap rate(s)
        """
        scalar_input = np.isscalar(T)
        T = np.atleast_1d(np.asarray(T, dtype=np.float64))

        result = np.zeros_like(T)

        for i, t_val in enumerate(T):
            if t_val <= 1e-10:
                result[i] = float(self.evaluate(1e-10))
                continue

            # Use fixed_quad for efficiency (Gauss-Legendre)
            integral, _ = fixed_quad(
                lambda x: self.evaluate((t_val / 2) * (x + 1)), -1, 1, n=15
            )
            integral *= t_val / 2  # Jacobian
            result[i] = integral / t_val

        if scalar_input:
            return float(result[0])
        return result

    def __call__(self, t: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """Shorthand for evaluate()."""
        return self.evaluate(t)


class ForwardVarianceCalculator:
    """
    Calculates forward variance and variance swap rates from market data.

    This class implements the Carr-Madan method for computing forward variance
    curves and variance swap rates. It loads market data, processes it using
    the mathematically rigorous ForwardVarianceCurve class, and saves results.

    Attributes:
        cfg (DictConfig): Configuration object containing parameters.
        logger (logging.Logger): Logger for tracking operations.
        input_path (Path): Path to input market data file.
        df (pd.DataFrame): Loaded market data.
        minimum_points (int): Minimum number of data points required.
        forward_variance_curve (ForwardVarianceCurve): Fitted forward variance curve.
        theta_curve (callable): Variance swap rate curve.
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
        self.df: Optional[pd.DataFrame] = None
        self.minimum_points = self.cfg.forward_variance.minimum_points

        # Get regularisation parameters from config, with sensible defaults
        self.n_knots = getattr(self.cfg.forward_variance, "n_knots", 10)
        self.regularisation_lambda = getattr(
            self.cfg.forward_variance, "regularisation_lambda", 0.01
        )

        self.forward_variance_curve: Optional[ForwardVarianceCurve] = None
        self._fvc_wrapper: Optional[Callable] = None  # Wrapper for compatibility
        self.theta_curve: Optional[Callable] = None
        self.output_path = Path(
            repo_root / self.cfg.forward_variance.output_path
        ).resolve()

        # Store fit diagnostics
        self.fit_result: Optional[ForwardVarianceFitResult] = None

    def load_data(self):
        """Load market data from the input path."""
        log_function_start("load_data")
        self.df = pd.read_csv(self.input_path)
        self.logger.info("Loaded market data from %s", self.input_path)
        self.logger.info("Data shape: %s", self.df.shape)
        self.logger.info(
            "Maturity range: %.4f to %.4f years",
            self.df["T_years"].min(),
            self.df["T_years"].max(),
        )
        log_function_end("load_data")

    def carr_madan_forward_variance(self):
        """
        Compute forward variance curve using ATM IV² with Carr-Madan fallback.

        This method uses a market-based approach to forward variance:
        
        1. Primary method (ATM IV²)**: Uses σ_ATM(T)² directly as the variance
           swap rate θ(T) at each maturity. This is a model-free approach that
           preserves the market's term structure shape and avoids numerical
           instabilities of Carr-Madan at short maturities.
           
        2. Fallback (Carr-Madan)**: For maturities without ATM IV data, uses
           the Carr-Madan variance swap formula, scaled to match nearby ATM IVs.

        After computing θ(T) at market maturities, we fit ξ(t) such that:

            θ(T) = (1/T) ∫₀ᵀ ξ(t) dt

        Using the exponential parameterization ξ(t) = exp(s(t)) for guaranteed positivity.
        """
        log_function_start("carr_madan_forward_variance")
        self.logger.info("Calculating forward variance using Carr-Madan method")

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        maturity_groups = self.df.groupby("T_years")

        var_swaps = []
        maturities = []
        n_options_at_maturity = []  # For weighting

        for maturity, group in maturity_groups:
            n_calls = len(group[group["cp_flag"] == "C"])
            n_puts = len(group[group["cp_flag"] == "P"])
            n_total = n_calls + n_puts

            if n_total < 20 or n_calls < 5 or n_puts < 5:
                self.logger.debug(
                    "Insufficient calls (%d) or puts (%d) for maturity %f, skipping...",
                    n_calls,
                    n_puts,
                    maturity,
                )
                continue

            forward = group["underlying_close"].iloc[0] * np.exp(
                group["risk_free_rate"].iloc[0] * maturity
            )
            theta = self._variance_swap_rate(group, forward, maturity)

            # Relaxed bounds - let the optimisation handle extreme values
            if 0.005 < theta < 25.0:  # More permissive than before
                maturities.append(maturity)
                var_swaps.append(theta)
                n_options_at_maturity.append(n_total)
            else:
                self.logger.warning(
                    "Variance swap rate %.4f at T=%.4f outside reasonable bounds, skipping",
                    theta,
                    maturity,
                )

        if len(maturities) < 5:
            self.logger.error(
                "Insufficient valid maturities (%d) for fitting.", len(maturities)
            )
            # Fallback: constant forward variance at typical equity vol
            self.logger.warning("Using constant fallback: ξ(t) = 0.225 (≈47.4%% vol)")
            self._fvc_wrapper = lambda t: np.full_like(np.atleast_1d(t), 0.225)
            self.theta_curve = lambda t: 0.225
            log_function_end("carr_madan_forward_variance")
            return

        T = np.array(maturities)
        theta_carr_madan = np.array(var_swaps)
        weights = np.array(n_options_at_maturity, dtype=np.float64)

        self.logger.info(
            "Fitting to %d maturities with Carr-Madan variance swaps in [%.4f, %.4f]",
            len(T),
            theta_carr_madan.min(),
            theta_carr_madan.max(),
        )

        # ----------------------------------------------------------------- #
        # HYBRID APPROACH: Scale Carr-Madan to match ATM IV² per maturity
        # ----------------------------------------------------------------- #
        # Carr-Madan gives variance swap rates, but can be miscalibrated.
        # ATM IV² is a reliable anchor point for each maturity.
        # 
        # Strategy:
        # 1. Compute ATM IV² for each maturity (interpolating if needed)
        # 2. Scale Carr-Madan θ(T) by the ratio ATM_IV²(T) / θ_CM(T)
        # 3. This preserves Carr-Madan's relative shape while anchoring to market
        # ----------------------------------------------------------------- #
        
        atm_iv_squared = np.zeros(len(T))
        atm_iv_available = np.zeros(len(T), dtype=bool)
        
        for i, T_i in enumerate(T):
            mat_df = self.df[np.abs(self.df["T_years"] - T_i) < 0.01]
            if len(mat_df) > 0:
                # Find ATM option (minimum |log-moneyness|)
                atm_idx = mat_df["log_moneyness"].abs().idxmin()
                atm_iv = mat_df.loc[atm_idx, "implied_volatility"]
                log_m = mat_df.loc[atm_idx, "log_moneyness"]
                
                # Use if within reasonable range of ATM (|log-moneyness| < 0.3)
                # and IV is valid. For options further from ATM, we interpolate
                # to get a better ATM estimate.
                if abs(log_m) < 0.3 and 0.05 < atm_iv < 3.0:
                    # If not close to ATM, try to interpolate between call and put
                    if abs(log_m) > 0.05:
                        # Find options on both sides of ATM for interpolation
                        otm_puts = mat_df[mat_df["log_moneyness"] < 0].sort_values("log_moneyness", ascending=False)
                        otm_calls = mat_df[mat_df["log_moneyness"] > 0].sort_values("log_moneyness")
                        
                        if len(otm_puts) > 0 and len(otm_calls) > 0:
                            # Linear interpolation to log_moneyness = 0
                            put_row = otm_puts.iloc[0]
                            call_row = otm_calls.iloc[0]
                            lm_put, iv_put = put_row["log_moneyness"], put_row["implied_volatility"]
                            lm_call, iv_call = call_row["log_moneyness"], call_row["implied_volatility"]
                            
                            # Interpolate IV at log_moneyness = 0
                            if lm_call != lm_put:
                                atm_iv = iv_put + (iv_call - iv_put) * (0 - lm_put) / (lm_call - lm_put)
                    
                    atm_iv_squared[i] = atm_iv ** 2
                    atm_iv_available[i] = True
        
        n_atm_available = np.sum(atm_iv_available)
        self.logger.info(
            "ATM IV² available for %d of %d maturities (%.0f%%)",
            n_atm_available, len(T), 100 * n_atm_available / len(T)
        )
        
        # Use Carr-Madan variance swap rates, scaled by median ATM IV² ratio
        # This preserves the term structure shape from Carr-Madan while 
        # correcting the overall level to match market ATM volatilities.
        theta = theta_carr_madan.copy()
        
        if n_atm_available >= 3:
            comparison_mask = atm_iv_available & (theta_carr_madan > 0)
            if np.any(comparison_mask):
                ratios = atm_iv_squared[comparison_mask] / theta_carr_madan[comparison_mask]
                median_ratio = np.median(ratios)
                self.logger.info(
                    "ATM IV² / Carr-Madan ratio: median=%.2f, range=[%.2f, %.2f]",
                    median_ratio, ratios.min(), ratios.max()
                )
                
                # Apply uniform scaling to preserve Carr-Madan term structure
                # Only scale if there's a significant discrepancy
                if median_ratio > 1.2 or median_ratio < 0.8:
                    self.logger.info(
                        "Scaling Carr-Madan variance swaps by %.2fx to match ATM IVs",
                        median_ratio
                    )
                    theta = theta_carr_madan * median_ratio
        
        self.logger.info(
            "Using ATM IV² for forward variance: θ range [%.4f, %.4f] (σ: %.1f%% to %.1f%%)",
            theta.min(), theta.max(), 
            np.sqrt(theta.min()) * 100, np.sqrt(theta.max()) * 100
        )

        # Create and fit the forward variance curve
        fvc = ForwardVarianceCurve(
            n_internal_knots=self.n_knots,
            regularisation_lambda=self.regularisation_lambda,
            extrapolation_method="constant",
        )

        try:
            self.fit_result = fvc.fit(T, theta, weights=weights)
        except Exception as e:
            self.logger.error("Forward variance fitting failed: %s", str(e))
            self.logger.warning("Using constant fallback")
            self._fvc_wrapper = lambda t: np.full_like(np.atleast_1d(t), np.mean(theta))
            self.theta_curve = lambda t: np.mean(theta)
            log_function_end("carr_madan_forward_variance")
            return

        self.forward_variance_curve = fvc

        # Wrapper for backward compatibility (returns array for any input)
        def xi_wrapper(t):
            return fvc.evaluate(np.atleast_1d(t))

        self._fvc_wrapper = xi_wrapper

        # Variance swap rate curve
        self.theta_curve = lambda t: fvc.get_variance_swap_rate(t)

        # Verify the fit
        fitted_theta = np.array([fvc.get_variance_swap_rate(t) for t in T])
        fit_error = np.sqrt(np.mean((np.log(fitted_theta) - np.log(theta)) ** 2))

        self.logger.info("Forward variance fit RMSE (log-space): %.6f", fit_error)

        # diagnostics logging
        xi_0 = fvc.evaluate(0.01)
        xi_half = fvc.evaluate(0.5)
        xi_1 = fvc.evaluate(1.0)
        xi_2 = fvc.evaluate(2.0) if T.max() > 2.0 else fvc.evaluate(T.max())

        self.logger.info("Forward variance diagnostics:")
        self.logger.info(
            "  ξ(0.01) = %.4f (%.1f%% instantaneous vol)", xi_0, np.sqrt(xi_0) * 100
        )
        self.logger.info(
            "  ξ(0.50) = %.4f (%.1f%% vol)", xi_half, np.sqrt(xi_half) * 100
        )
        self.logger.info("  ξ(1.00) = %.4f (%.1f%% vol)", xi_1, np.sqrt(xi_1) * 100)
        self.logger.info("  ξ(max)  = %.4f (%.1f%% vol)", xi_2, np.sqrt(xi_2) * 100)

        # Verify positivity (should always pass with exponential parameterization)
        t_test = np.linspace(0.01, T.max() * 1.2, 100)
        xi_test = fvc.evaluate(t_test)
        if np.any(xi_test <= 0):
            self.logger.error(
                "CRITICAL: Forward variance not positive! Min = %.6f", xi_test.min()
            )
        else:
            self.logger.info("Positivity verified: min(ξ) = %.6f", xi_test.min())

        log_function_end("carr_madan_forward_variance")

    def _variance_swap_rate(
        self, df: pd.DataFrame, forward_price: float, maturity: float
    ) -> float:
        """
        Compute variance swap rate using Carr-Madan formula.

        θ(T) = (2e^(rT)/T) ∫ [P(K)/K² dK for K<F] + [C(K)/K² dK for K>F]

        Parameters
        ----------
        df : pd.DataFrame
            Option data for this maturity
        forward_price : float
            Forward price F = S * exp(rT)
        maturity : float
            Time to maturity in years

        Returns
        -------
        float
            Variance swap rate θ(T)
        """
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
            self.logger.debug(
                "Too few OTM options (%d) for maturity=%f; using ATM approximation",
                total_otm,
                maturity,
            )
            return self._atm_approx(df, maturity)

        var_contribution = 0.0

        # OTM calls contribution: ∫_{K>F} C(K)/K² dK
        if not otm_calls.empty:
            strikes = otm_calls["strike_price"].to_numpy()
            prices = otm_calls["mid"].to_numpy()
            idx = np.argsort(strikes)
            strikes = strikes[idx]
            prices = prices[idx]

            # Filter out non-positive prices
            valid = prices > 0
            if np.any(valid):
                strikes = strikes[valid]
                prices = prices[valid]
                integrand = prices / strikes**2
                if len(strikes) > 1:
                    var_contribution += np.trapezoid(integrand, strikes)
                elif len(strikes) == 1:
                    var_contribution += integrand[0] * (strikes[0] * 0.01)

        # OTM puts contribution: ∫_{K<F} P(K)/K² dK
        if not otm_puts.empty:
            strikes = otm_puts["strike_price"].to_numpy()
            prices = otm_puts["mid"].to_numpy()
            idx = np.argsort(strikes)
            strikes = strikes[idx]
            prices = prices[idx]

            valid = prices > 0
            if np.any(valid):
                strikes = strikes[valid]
                prices = prices[valid]
                integrand = prices / strikes**2
                if len(strikes) > 1:
                    var_contribution += np.trapezoid(integrand, strikes)
                elif len(strikes) == 1:
                    var_contribution += integrand[0] * (strikes[0] * 0.01)

        if var_contribution <= 0:
            self.logger.debug(
                "Non-positive variance contribution; using ATM approximation for maturity=%f",
                maturity,
            )
            return self._atm_approx(df, maturity)

        r = df["risk_free_rate"].iloc[0]
        theta = (2 * np.exp(r * maturity) * var_contribution) / maturity

        return max(theta, 0.0)

    def _atm_approx(self, df: pd.DataFrame, maturity: float) -> float:
        """
        Approximate variance swap rate using ATM implied volatility.

        When Carr-Madan integration fails, use θ ≈ σ²_ATM as fallback.
        """
        _unused = maturity
        # Find ATM option (minimum log-moneyness)
        atm_idx = df["log_moneyness"].abs().idxmin()
        iv_atm = df.loc[atm_idx, "implied_volatility"]
        return iv_atm**2

    def save_data(self):
        """
        Save forward variance and variance swap rate data to CSV file.

        Creates a DataFrame with maturity, forward variance, and variance swap rate
        values, and saves it to the configured output path.
        """
        log_function_start("save_data")

        if self._fvc_wrapper is None:
            self.logger.error("Cannot save: forward variance curve not computed.")
            log_function_end("save_data")
            return

        max_maturity = float(self.df["T_years"].max())
        t_plot = np.linspace(0.01, max_maturity, 100)
        forward_variance = self._fvc_wrapper(t_plot)

        # Ensure it's a numpy array
        forward_variance = np.asarray(forward_variance).flatten()

        # Compute variance swap rates
        theta_vol = np.zeros_like(t_plot)
        if self.theta_curve is not None:
            for i, t in enumerate(t_plot):
                try:
                    theta_var = float(self.theta_curve(t))
                    theta_vol[i] = np.sqrt(max(theta_var, 0.0))
                except Exception:
                    theta_vol[i] = np.nan

        output_df = pd.DataFrame(
            {
                "maturity_years": t_plot,
                "forward_variance": forward_variance,
                "variance_swap_rate": theta_vol,
            }
        )

        self.output_path.mkdir(parents=True, exist_ok=True)
        output_file = self.output_path / "forward_variance_data.csv"
        output_df.to_csv(output_file, index=False)
        self.logger.info("Saved forward variance and theta curves to %s", output_file)

        # Also save fit diagnostics if available
        if self.fit_result is not None:
            diagnostics = {
                "success": self.fit_result.success,
                "rmse_log_space": self.fit_result.rmse_log_space,
                "n_iterations": self.fit_result.n_iterations,
                "message": self.fit_result.message,
                "n_knots": len(self.fit_result.knots),
                "regularisation_lambda": self.regularisation_lambda,
            }
            import json

            diagnostics_file = self.output_path / "forward_variance_diagnostics.json"
            with open(diagnostics_file, "w") as f:
                json.dump(diagnostics, f, indent=2)
            self.logger.info("Saved fit diagnostics to %s", diagnostics_file)

        log_function_end("save_data")

    def plot_variance(self):
        """
        Plot forward variance and variance swap rate curves.

        Generates a dual-axis plot with forward variance and variance swap rates,
        saving the figure to the output path.
        """
        log_function_start("plot_variance")

        if self._fvc_wrapper is None or self.df is None:
            self.logger.warning("Cannot plot: forward variance not computed")
            log_function_end("plot_variance")
            return

        fig, ax1 = plt.subplots(figsize=(12, 6))

        max_maturity = float(self.df["T_years"].max())
        t_plot = np.linspace(0.01, max_maturity, 100)
        forward_variance = np.asarray(self._fvc_wrapper(t_plot)).flatten()

        ax1.plot(
            t_plot,
            forward_variance,
            label="Forward Variance ξ(t)",
            color="blue",
            linewidth=2,
        )
        ax1.set_xlabel("Maturity (Years)", fontsize=12)
        ax1.set_ylabel("Forward Variance", color="blue", fontsize=12)
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        if self.theta_curve is not None:
            theta_plot = []
            for t in t_plot:
                try:
                    theta_plot.append(float(self.theta_curve(t)))
                except Exception:
                    theta_plot.append(np.nan)
            theta_plot = np.array(theta_plot)
            ax2.plot(
                t_plot,
                theta_plot,
                label="Variance Swap θ(t)",
                color="green",
                linestyle="--",
                linewidth=2,
            )
        ax2.set_ylabel("Variance Swap Rate", color="green", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="green")

        # Add volatility scale on right side
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.12))
        ax3.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1])
        ax3.set_ylabel("Implied Vol (%)", color="gray", fontsize=12)

        fig.suptitle("Forward Variance Curve (Exponential Spline Fit)", fontsize=14)
        fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.95))
        plt.tight_layout()

        self.output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            self.output_path / "forward_variance.svg", format="svg", bbox_inches="tight"
        )
        plt.savefig(
            self.output_path / "forward_variance.png",
            format="png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

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
    logger.info(
        "Starting forward variance calculation..."
    )

    calculator = ForwardVarianceCalculator(cfg)
    calculator.load_data()
    calculator.carr_madan_forward_variance()
    calculator.save_data()
    calculator.plot_variance()

    logger.info("Forward variance calculation completed.")


if __name__ == "__main__":
    main()
