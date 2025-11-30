"""Core pricing and calibration engine for the Rough Bergomi model.

The :class:`RoughBergomiEngine` orchestrates:
- Loading of market data (forward variance curve, implied volatility surface, rates)
- Interpolation of zero yields and instantaneous short rates
- High-performance Monte-Carlo pricing with antithetic variates
- Global Vega-weighted least-squares calibration of (η, H, ρ)

All heavy numerical kernels are implemented in separate modules and imported here.
"""
from typing import Optional
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev
from scipy.optimize import least_squares
from numpy.random import default_rng
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path


from rbergomi.convolution import precompute_convolution_matrix
from rbergomi.simulation import simulate_rbergomi_paths
from rbergomi.black_scholes import implied_volatility, compute_vega_scalar
from src.rbergomi.calibration_result import CalibrationResult
from scripts.logging_config import get_logger


class RoughBergomiEngine:
    """
    Core engine for Rough Bergomi pricing and calibration.

    Handles:
        - Loading of forward variance curve ξ_t(T)
        - Loading of implied volatility surface
        - Interpolation of risk-free rates and short rates
        - Monte-Carlo pricing with antithetic variates
        - Global least-squares calibration with regularisation
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the engine from Hydra configuration.

        Loads all required market data (forward variance, IV surface, rates)
        and builds interpolators.
        """
        self.cfg = cfg
        self.logger = get_logger("RoughBergomiEngine")
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent.parent
        f_variance_path = Path(repo_root / self.cfg.f_variance.output_path).resolve()
        self.df = pd.read_csv(f_variance_path)

        # --------------------- Forward variance curve --------------------- #
        self.forward_variance_func = interp1d(
            self.df["maturity_years"],
            self.df["forward_variance"],
            bounds_error=False,
            fill_value=(
                float(self.df["forward_variance"].iloc[0]),
                float(self.df["forward_variance"].iloc[-1]),
            ),
        )

        # --------------------- Implied volatility surface --------------------- #
        iv_surface_path = Path(repo_root / self.cfg.iv_surface.surface_path).resolve()
        self.iv_surface = np.load(iv_surface_path)
        if np.any(np.isnan(self.iv_surface)) or np.any(self.iv_surface <= 0):
            raise ValueError("IV surface contains NaN or non-positive values")

        maturities_path = Path(
            repo_root / self.cfg.iv_surface.maturities_path
        ).resolve()
        self.maturities = np.load(maturities_path)
        log_moneyness_path = Path(
            repo_root / self.cfg.iv_surface.log_moneyness_path
        ).resolve()
        self.log_moneyness = np.load(log_moneyness_path)
        if np.any(np.isnan(self.log_moneyness)):
            raise ValueError("Log moneyness contain NaN values")
        self.strikes = np.exp(self.log_moneyness)

        # --------------------- Risk-free rate curve --------------------- #
        rf_path = self.cfg.iv_surface.risk_free_rate_path
        self.risk_free_rate_path = Path(repo_root / rf_path).resolve()
        self.risk_free_rate = np.load(self.risk_free_rate_path)

        if np.any(np.isnan(self.maturities)) or np.any(self.maturities <= 0):
            raise ValueError("Maturities contain NaN or non-positive values")

        risk_free_rates = self.risk_free_rate.copy()
        # Ensure spline works at t=0
        if self.maturities[0] > 0:
            self.maturities = np.insert(self.maturities, 0, 0)
            risk_free_rates = np.insert(self.risk_free_rate, 0, self.risk_free_rate[0])
        self.yield_spl = splrep(self.maturities, risk_free_rates, s=0, k=3)

        if (
            self.df["maturity_years"].isnull().any()
            or self.df["forward_variance"].isnull().any()
        ):
            raise ValueError("Forward variance data contains NaN values")
        if (self.df["forward_variance"] <= 0).any():
            raise ValueError("Forward variance contains non-positive values")

        start = self.df["maturity_years"].min()
        end = self.df["maturity_years"].max()
        self.logger.info("Forward variance data range: %s to %s", start, end)

    # --------------------------------------------------------------------- #
    # Yield & short rate interpolation
    # --------------------------------------------------------------------- #

    def get_yield(self, t: float | np.ndarray) -> float | np.ndarray:
        """Zero-coupon yield Y(t) = -ln(P(0,t))/t"""
        res = splev(t, self.yield_spl, ext=0)
        res_arr = np.asarray(res)
        if np.ndim(res_arr) == 0:
            return float(res_arr)
        return res_arr

    def get_short_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Instantaneous short rate r(t) = Y(t) + t * dY/dt(t)
        Computed via spline derivative.
        """
        if np.isscalar(t):
            # Handle t=0 case directly
            if t == 0.0:
                rate = float(np.atleast_1d(splev(1e-10, self.yield_spl, ext=0))[0])
                return rate
            try:
                # Convert to a numpy scalar/array first and extract a Python scalar
                t_arr = np.asarray(t)
                # If complex-valued, use the real part to avoid float(complex) errors
                if np.iscomplexobj(t_arr):
                    t_val = float(np.real(t_arr).item())
                else:
                    t_val = float(t_arr.item())
            except (TypeError, ValueError, IndexError, AttributeError):
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

    # --------------------------------------------------------------------- #
    # Monte-Carlo pricing
    # --------------------------------------------------------------------- #

    def price_options(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        params: tuple[float, float, float] | np.ndarray,
        n_paths: int = 50000,
        n_steps: int = 512,
        return_iv: bool = False,
        option_type: str = "call",
        return_terminal_paths: bool = False,
        precomputed_g1: np.ndarray | None = None,
        precomputed_z: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Price a set of European options using Monte-Carlo under rBergomi dynamics.

        Parameters
        ----------
        strikes, maturities : np.ndarray
            Grid of strikes and maturities to price
        params : tuple/array
            (eta, H, rho)
        n_paths, n_steps : int
            Monte-Carlo resolution
        return_iv : bool
            Return implied volatilities instead of prices
        option_type : {'call', 'put'}
        return_terminal_paths : bool
            Return raw terminal stock levels (used internally by calibration)

        Returns
        -------
        np.ndarray
            Prices or IVs on the requested grid
        """
        eta, hurst, rho = params
        alpha = hurst - 0.5
        n_steps = (
            8196 if max(maturities) < 0.1 else n_steps
        )  # finer grid for short maturities

        max_t = max(maturities)
        dt = max_t / n_steps
        K = precompute_convolution_matrix(alpha, n_steps)

        t = np.linspace(0, max_t, n_steps + 1)
        xi = self.forward_variance_func(t)
        short_rates = np.asarray(self.get_short_rate(t), dtype=float)

        if np.any(xi <= 0) or np.any(np.isnan(xi)):
            self.logger.error("Invalid forward variance values: %s", xi)
            raise ValueError("Forward variance contains non-positive or NaN values")
        self.logger.info(
            "Forward variance shape: %s, values: %s...",
            xi.shape,
            xi[:5],
        )

        # --------------------- Antithetic variates --------------------- #
        # Optionally reuse precomputed normals for calibration stability
        if precomputed_g1 is None or precomputed_z is None:
            seed = self.cfg.seed
            rng = default_rng(seed)
            n_base = n_paths // 2

            all_g1_base = rng.standard_normal((n_base, n_steps))
            all_g1 = np.vstack([all_g1_base, -all_g1_base])[:n_paths, :]

            all_z_base = rng.standard_normal((n_base, n_steps))
            all_z = np.vstack([all_z_base, -all_z_base])[:n_paths, :]
        else:
            all_g1 = precomputed_g1
            all_z = precomputed_z

        # --------------------- Simulation --------------------- #
        s_paths, variance_paths = simulate_rbergomi_paths(
            n_paths, n_steps, max_t, eta, hurst, rho, xi, all_g1, all_z, short_rates, K
        )
        if np.any(np.isnan(s_paths)) or np.any(s_paths <= 0):
            self.logger.error("Stock paths contain NaN or non-positive values")
            raise ValueError("Invalid stock paths")
        if np.any(np.isnan(variance_paths)) or np.any(variance_paths <= 0):
            self.logger.error("Variance paths contain NaN or non-positive values")
            raise ValueError("Invalid variance paths")

        # --------------------- Pricing --------------------- #
        indices = np.round(maturities / dt).astype(np.int64)
        s_t_all = s_paths[:, indices]  # (n_paths, n_maturities)

        yields = np.asarray(self.get_yield(maturities), dtype=float)
        discounts = np.exp(-yields * maturities)

        s_t = s_t_all
        discounts = discounts[:, None]

        if return_terminal_paths:
            # Shape (n_paths, n_maturities, 1)
            return s_t[:, :, np.newaxis]

        # Payoffs on already-discounted paths
        if option_type == "call":
            payoffs = np.maximum(s_t[:, :, None] - strikes[None, None, :], 0)
        else:  # put
            payoffs = np.maximum(strikes[None, None, :] - s_t[:, :, None], 0)

        prices = np.mean(payoffs, axis=0) * discounts  # discount after the nonlinearity
        prices = np.clip(prices, 1e-12, None)

        if return_iv:
            ivs = np.zeros(prices.shape)
            for m, maturity in enumerate(maturities):
                r = float(yields[m])
                for k, strike in enumerate(strikes):
                    ivs[m, k] = (
                        implied_volatility(prices[m, k], 1.0, strike, maturity, r=r)
                        if prices[m, k] > 0
                        else np.nan
                    )
            return ivs
        return prices

    # --------------------------------------------------------------------- #
    # Calibration
    # --------------------------------------------------------------------- #

    def calibration(
        self,
        target_call_prices: Optional[np.ndarray],
        target_put_prices: Optional[np.ndarray],
        strikes: np.ndarray,
        maturities: np.ndarray,
        initial_params: tuple[float, float, float] | np.ndarray,
    ) -> "CalibrationResult":
        """
        Calibrate (η, H, ρ) to market IV surface using Vega-weighted least squares.

        Mathematically rigorous calibration that minimises:

            Σᵢⱼ Vegaᵢⱼ * [(IV_model - IV_market) / IV_market]²

        Subject to parameter bounds:
            η ∈ [0.1, 10.0]      - Vol-of-vol parameter
            H ∈ [0.005, 0.20]    - Hurst parameter (rough regime)
            ρ ∈ [-1.0, -0.10]    - Spot-vol correlation (negative for equity)

        Key improvements over previous implementation:
        1. Vega weighting: Points with higher Vega (more sensitive to vol) get more weight
        2. No arbitrary parameter penalties: Let bounds handle constraints
        3. No smoothness penalty on errors: Forward variance handles smoothness
        4. Pure IV-space calibration: More stable than price-space

        Parameters
        ----------
        target_call_prices, target_put_prices : Optional[np.ndarray]
            Market call and put option prices (used only for fallback)
        strikes, maturities : np.ndarray
            Strike and maturity grids
        initial_params : tuple or np.ndarray
            Initial guess for [eta, hurst, rho]

        Returns
        -------
        CalibrationResult
            Object containing optimal parameters, fitted IVs, and fit metrics
        """
        # Filter very short maturities (numerically unstable)
        maturities_filter = self.cfg.calibration.maturities_filter
        valid_indices = maturities >= maturities_filter
        filtered_maturities = maturities[valid_indices]
        filtered_strikes = strikes

        # Market IVs for this calibration
        market_ivs = self.iv_surface[valid_indices]

        # Compute yields for each maturity
        yields = np.asarray(self.get_yield(filtered_maturities), dtype=float)

        self.logger.info(
            "Calibrating to %d maturities × %d strikes = %d points",
            len(filtered_maturities),
            len(filtered_strikes),
            len(filtered_maturities) * len(filtered_strikes),
        )

        # ----------------------------------------------------------------- #
        # Compute Vega weights for all calibration points
        # ----------------------------------------------------------------- #
        vegas = np.zeros_like(market_ivs)
        for m, (T, r) in enumerate(zip(filtered_maturities, yields)):
            for k, K in enumerate(filtered_strikes):
                if np.isfinite(market_ivs[m, k]) and market_ivs[m, k] > 0:
                    # Use S=1.0 for normalized grid
                    vegas[m, k] = compute_vega_scalar(1.0, K, T, r, market_ivs[m, k])

        # Normalize Vegas to have mean = 1 for numerical stability
        vega_mask = vegas > 0
        if np.any(vega_mask):
            vega_mean = np.mean(vegas[vega_mask])
            if vega_mean > 0:
                vegas = vegas / vega_mean
            else:
                vegas = np.ones_like(vegas)
        else:
            self.logger.warning("No valid Vega weights; using uniform weighting")
            vegas = np.ones_like(market_ivs)

        n_valid = np.sum((market_ivs > 0) & np.isfinite(market_ivs) & (vegas > 0))
        self.logger.info("Number of valid calibration points: %d", n_valid)

        # ----------------------------------------------------------------- #
        # Pre-generate shared randomness for deterministic calibration
        # ----------------------------------------------------------------- #
        bounds = ([0.1, 0.005, -1.0], [10.0, 0.20, -0.01])  # η, H, ρ
        n_paths = self.cfg.calibration.n_paths
        n_steps = self.cfg.calibration.n_steps
        seed = self.cfg.seed

        rng_cal = default_rng(seed)
        n_base_cal = n_paths // 2
        g1_base_cal = rng_cal.standard_normal((n_base_cal, n_steps))
        all_g1_cal = np.vstack([g1_base_cal, -g1_base_cal])[:n_paths, :]
        z_base_cal = rng_cal.standard_normal((n_base_cal, n_steps))
        all_z_cal = np.vstack([z_base_cal, -z_base_cal])[:n_paths, :]

        # ----------------------------------------------------------------- #
        # Vega-weighted IV-space objective function
        # ----------------------------------------------------------------- #
        # Pre-compute the fixed calibration mask (must be constant across calls)
        fixed_calibration_mask = (
            (market_ivs > 0) & np.isfinite(market_ivs) & (vegas > 0)
        )
        n_calibration_points = np.sum(fixed_calibration_mask)
        self.logger.info("Using %d fixed calibration points", n_calibration_points)

        def objective(params):
            """
            Compute Vega-weighted relative IV errors.

            Returns vector of weighted errors for least_squares optimiser.
            The optimiser will minimize sum(errors²), so we return:
                sqrt(vega) * (IV_model - IV_market) / IV_market

            This gives us Vega-weighted relative IV errors when squared.

            IMPORTANT: Must return same-sized array for all parameter values!
            """
            eta, hurst, rho = params

            # Single MC simulation with shared randomness
            s_t = self.price_options(
                filtered_strikes,
                filtered_maturities,
                params,
                n_paths=n_paths,
                n_steps=n_steps,
                return_terminal_paths=True,
                precomputed_g1=all_g1_cal,
                precomputed_z=all_z_cal,
            ).squeeze(-1)  # shape: (n_paths, n_maturities)

            discounts = np.exp(-yields * filtered_maturities)[:, None]

            # Compute model call prices
            call_payoffs = np.maximum(
                s_t[:, :, None] - filtered_strikes[None, None, :], 0
            )
            model_call_prices = np.mean(call_payoffs, axis=0) * discounts

            # Convert model prices to implied volatilities
            model_ivs = np.full_like(model_call_prices, np.nan)
            for m, (T, r) in enumerate(zip(filtered_maturities, yields)):
                for k, K in enumerate(filtered_strikes):
                    price = model_call_prices[m, k]
                    if price > 1e-12:
                        try:
                            model_ivs[m, k] = implied_volatility(
                                price, 1.0, K, T, r=r, option_type="call"
                            )
                        except Exception:
                            model_ivs[m, k] = np.nan

            # Initialise residuals array with fixed size
            residuals = np.zeros(n_calibration_points)

            # Get values at calibration points
            market_ivs_valid = market_ivs[fixed_calibration_mask]
            model_ivs_valid = model_ivs[fixed_calibration_mask]
            vega_weights = np.sqrt(vegas[fixed_calibration_mask])

            # Handle invalid model IVs by assigning large residual
            valid_model = np.isfinite(model_ivs_valid) & (model_ivs_valid > 0)

            # Where model is valid: compute relative error
            residuals[valid_model] = vega_weights[valid_model] * (
                (model_ivs_valid[valid_model] - market_ivs_valid[valid_model])
                / market_ivs_valid[valid_model]
            )

            # Where model is invalid: assign large penalty (but not too large to cause overflow)
            residuals[~valid_model] = 10.0  # 1000% relative error

            return residuals

        # ----------------------------------------------------------------- #
        # Run optimisation
        # ----------------------------------------------------------------- #
        self.logger.info(
            "Starting calibration with initial params: eta=%.3f, H=%.4f, rho=%.3f",
            initial_params[0],
            initial_params[1],
            initial_params[2],
        )

        res = least_squares(
            objective,
            np.asarray(initial_params),
            bounds=bounds,
            method="trf",
            max_nfev=5000,
            ftol=1e-6,
            gtol=1e-6,
            xtol=1e-8,
            loss="soft_l1",  # Robust to outliers
            verbose=2,
        )

        optimal_params = res.x
        self.logger.info(
            "Optimal parameters: η=%.4f, H=%.4f, ρ=%.4f",
            optimal_params[0],
            optimal_params[1],
            optimal_params[2],
        )

        # ----------------------------------------------------------------- #
        # Compute fitted IVs with optimal parameters
        # ----------------------------------------------------------------- #
        s_t_final = self.price_options(
            filtered_strikes,
            filtered_maturities,
            optimal_params,
            n_paths=self.cfg.final_pricing.n_paths,
            n_steps=self.cfg.final_pricing.n_steps,
            return_terminal_paths=True,
        ).squeeze(-1)

        discounts = np.exp(-yields * filtered_maturities)[:, None]

        call_payoffs = np.maximum(
            s_t_final[:, :, None] - filtered_strikes[None, None, :], 0
        )
        fitted_call_prices = np.mean(call_payoffs, axis=0) * discounts
        fitted_ivs = np.full_like(fitted_call_prices, np.nan)

        for m, maturity in enumerate(filtered_maturities):
            r = float(yields[m])
            for k, strike in enumerate(filtered_strikes):
                price = fitted_call_prices[m, k]
                if price > 0 and np.isfinite(price):
                    try:
                        fitted_ivs[m, k] = implied_volatility(
                            price, 1.0, strike, maturity, r=r, option_type="call"
                        )
                    except Exception:
                        fitted_ivs[m, k] = np.nan

        # ----------------------------------------------------------------- #
        # Build calibration result
        # ----------------------------------------------------------------- #
        result = CalibrationResult(self.cfg)
        result.optimal_params = {
            "eta": float(optimal_params[0]),
            "hurst": float(optimal_params[1]),
            "rho": float(optimal_params[2]),
        }
        result.fitted_ivs = fitted_ivs
        result.market_ivs = market_ivs

        # Compute RMSE (unweighted, for reporting)
        if result.fitted_ivs is not None and result.market_ivs is not None:
            valid_mask = np.isfinite(result.fitted_ivs) & np.isfinite(result.market_ivs)
            if np.any(valid_mask):
                result.rmse = float(
                    np.sqrt(
                        np.mean(
                            (
                                result.fitted_ivs[valid_mask]
                                - result.market_ivs[valid_mask]
                            )
                            ** 2
                        )
                    )
                )

                # Also compute weighted RMSE for comparison
                if np.any(valid_mask & (vegas > 0)):
                    weighted_sq_errors = (
                        vegas[valid_mask]
                        * (
                            (
                                result.fitted_ivs[valid_mask]
                                - result.market_ivs[valid_mask]
                            )
                            / result.market_ivs[valid_mask]
                        )
                        ** 2
                    )
                    weighted_rmse = float(np.sqrt(np.mean(weighted_sq_errors)))
                    self.logger.info(
                        "Vega-weighted RMSE (relative): %.6f", weighted_rmse
                    )
            else:
                result.rmse = np.nan
                self.logger.warning("No valid IVs for RMSE calculation")
        else:
            result.rmse = np.nan
            self.logger.warning(
                "Cannot compute RMSE because fitted_ivs or market_ivs is None"
            )

        self.logger.info("Calibration RMSE (absolute IV): %.6f", result.rmse)

        result.convergence_info = res
        return result
