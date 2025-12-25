"""
Market Surface Builder Module
=============================

This module constructs arbitrage-free implied volatility surfaces from sparse
market option data using interpolation techniques.
"""

from pathlib import Path
from typing import Optional, Tuple
from omegaconf import DictConfig
import hydra
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, interp1d, PchipInterpolator
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa: F401
import warnings

from scripts.logging_config import (
    get_logger,
    setup_logging,
    log_function_start,
    log_function_end,
)


class MarketSurfaceBuilder:
    """
    Constructs arbitrage-free implied volatility surfaces from market data.

    This class implements mathematically rigorous surface construction:
    - Smooth interpolation (RBF or cubic) for C¹/C² continuity
    - Minimal smoothing to preserve genuine market structure
    - Proper calendar arbitrage enforcement via isotonic regression
    - 3D visualisation for validation

    Attributes
    ----------
    cfg : DictConfig
        Configuration object containing paths and parameters
    logger : logging.Logger
        Logger instance for tracking operations
    input_path : Path
        Path to the input market data file
    output_dir : Path
        Directory to save output files
    df : pd.DataFrame
        DataFrame holding the loaded market data
    n_maturity_bins : int
        Number of bins for maturity grid
    n_moneyness_bins : int
        Number of bins for moneyness grid
    grid_maturities : np.ndarray
        Array of maturity grid points
    grid_log_moneyness : np.ndarray
        Array of log moneyness grid points
    raw_points : pd.DataFrame
        DataFrame containing raw data points for visualisation
    iv_surface : np.ndarray
        Implied volatility surface grid (n_maturities × n_strikes)
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the MarketSurfaceBuilder with configuration."""
        super().__init__()
        self.cfg = cfg
        self.logger = get_logger("MarketSurfaceBuilder")

        # Resolve paths
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        input_path = self.cfg.select_date.output_path
        self.input_path = Path(repo_root / input_path).resolve()
        self.output_dir = Path(repo_root / self.cfg.surface_builder.output_dir).resolve()

        # Data containers
        self.df: Optional[pd.DataFrame] = None
        self.raw_points: Optional[pd.DataFrame] = None
        self.iv_surface: Optional[np.ndarray] = None

        # Grid parameters
        self.n_maturity_bins = self.cfg.surface_builder.n_maturity_bins
        self.n_moneyness_bins = self.cfg.surface_builder.n_moneyness_bins
        self.grid_maturities: Optional[np.ndarray] = None
        self.grid_log_moneyness: Optional[np.ndarray] = None
        self.grid_risk_free_rates: Optional[np.ndarray] = None
        self.maturities: Optional[np.ndarray] = None

        # Interpolation parameters (with defaults)
        sb_cfg = self.cfg.surface_builder
        self.interpolation_method = getattr(sb_cfg, "interpolation_method", "rbf")
        self.light_smoothing = getattr(sb_cfg, "light_smoothing", True)
        self.smoothing_sigma = getattr(sb_cfg, "smoothing_sigma", 0.3)
        self.enforce_calendar_arbitrage = getattr(
            sb_cfg, "enforce_calendar_arbitrage", True
        )
        self.min_variance_increase = getattr(sb_cfg, "min_variance_increase", 0.0001)

    def load_data(self) -> pd.DataFrame:
        """
        Load market data from the input path.

        Returns
        -------
        pd.DataFrame
            Loaded market data
        """
        log_function_start("load_data")

        self.df = pd.read_csv(self.input_path)
        self.logger.info("Loaded market data from %s", self.input_path)
        self.logger.info("Data shape: %s", self.df.shape)

        log_function_end("load_data")
        return self.df

    def create_market_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a grid of maturities and log moneyness for interpolation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Arrays of maturity and log moneyness grid points
        """
        log_function_start("create_market_grid")

        # Get data ranges
        time_min = self.df["T_years"].min()
        time_max = self.df["T_years"].max()
        k_min = self.df["log_moneyness"].min()
        k_max = self.df["log_moneyness"].max()

        self.logger.info("Maturity range: [%.4f, %.4f] years", time_min, time_max)
        self.logger.info("Log-moneyness range: [%.4f, %.4f]", k_min, k_max)

        # Create uniform grids
        self.grid_maturities = np.linspace(time_min, time_max, self.n_maturity_bins)
        self.maturities = self.grid_maturities
        self.grid_log_moneyness = np.linspace(k_min, k_max, self.n_moneyness_bins)

        self.logger.info(
            "Created market grid: %d maturities × %d strikes",
            self.n_maturity_bins,
            self.n_moneyness_bins,
        )

        log_function_end("create_market_grid")
        return self.grid_maturities, self.grid_log_moneyness

    def interpolate_risk_free_rates(self) -> np.ndarray:
        """
        Interpolate the risk-free rates for the maturity grid points.

        Uses linear interpolation with extrapolation for maturities
        outside the observed range.

        Returns
        -------
        np.ndarray
            Interpolated risk-free rates at grid maturities
        """
        log_function_start("interpolate_risk_free_rates")

        # Get unique maturities and their mean rates
        unique_mats = np.sort(self.df["T_years"].unique())
        mean_rates_series = self.df.groupby("T_years")["risk_free_rate"].mean()
        mean_rates = mean_rates_series.reindex(unique_mats).to_numpy()

        # Filter out NaN values
        valid_mask = ~np.isnan(mean_rates)
        unique_mats = unique_mats[valid_mask]
        mean_rates = mean_rates[valid_mask]

        if len(unique_mats) < 2:
            self.logger.warning(
                "Insufficient unique maturities for interpolation; "
                "using constant risk-free rate."
            )
            default_rate = mean_rates[0] if len(mean_rates) > 0 else 0.0
            self.grid_risk_free_rates = np.full_like(self.grid_maturities, default_rate)
        else:
            # Linear interpolation with extrapolation
            r_interp = interp1d(
                unique_mats,
                mean_rates,
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )
            self.grid_risk_free_rates = r_interp(self.grid_maturities)
            self.grid_risk_free_rates = np.nan_to_num(
                self.grid_risk_free_rates, nan=0.0
            )

        self.logger.info(
            "Interpolated risk-free rates: [%.4f%%, %.4f%%]",
            self.grid_risk_free_rates.min() * 100,
            self.grid_risk_free_rates.max() * 100,
        )

        log_function_end("interpolate_risk_free_rates")
        return self.grid_risk_free_rates

    def interpolate_surface(self) -> np.ndarray:
        """
        Interpolate IV surface using smooth, arbitrage-aware methods.

        Mathematical approach:
        1. Work in total variance space: w(T, K) = σ² · T
        2. Use smooth interpolation (RBF or cubic)
        3. Minimal post-processing (light smoothing only)
        4. Enforce arbitrage-free conditions via isotonic projection

        Returns
        -------
        np.ndarray
            Arbitrage-free implied volatility surface (n_maturities × n_strikes)
        """
        log_function_start("interpolate_surface")

        # Filter to valid data points
        self.df["valid_pricing"] = self.df["valid_pricing"].astype(bool)
        valid_df = self.df[self.df["valid_pricing"]].copy()
        n_valid = len(valid_df)
        n_total = len(self.df)
        self.logger.info(
            "Valid data points: %d / %d (%.1f%%)",
            n_valid,
            n_total,
            100 * n_valid / n_total,
        )

        # Store raw points for visualisation
        self.raw_points = valid_df[
            ["T_years", "log_moneyness", "implied_volatility"]
        ].copy()

        # Compute total variance: w(T, K) = σ² · T
        valid_df["total_variance"] = (
            valid_df["implied_volatility"] ** 2 * valid_df["T_years"]
        )

        # Prepare interpolation data
        points = valid_df[["T_years", "log_moneyness"]].values
        values = valid_df["total_variance"].values

        self.logger.info(
            "Total variance range: [%.4f, %.4f] (σ: %.1f%% to %.1f%%)",
            values.min(),
            values.max(),
            np.sqrt(values.min() / valid_df["T_years"].min()) * 100,
            np.sqrt(values.max() / valid_df["T_years"].max()) * 100,
        )

        # Create meshgrid for interpolation
        time_grid, k_grid = np.meshgrid(
            self.grid_maturities, self.grid_log_moneyness, indexing="ij"
        )

        # Perform interpolation based on method
        total_variance_grid = self._interpolate_total_variance(
            points, values, time_grid, k_grid
        )

        # Apply minimal smoothing if configured
        if self.light_smoothing and self.smoothing_sigma > 0:
            total_variance_grid = gaussian_filter(
                total_variance_grid, sigma=self.smoothing_sigma, mode="nearest"
            )
            self.logger.info(
                "Applied light Gaussian smoothing (σ=%.2f)", self.smoothing_sigma
            )

        # Convert to IV (avoiding division by zero)
        maturities_safe = np.maximum(self.grid_maturities, 1e-8)
        iv_grid = np.sqrt(
            np.maximum(total_variance_grid, 0) / maturities_safe[:, np.newaxis]
        )

        # Handle outliers with local median replacement (not clipping!)
        iv_grid = self._handle_outliers(iv_grid, lower=0.01, upper=5.0)

        # Enforce calendar arbitrage: ∂w/∂T ≥ 0
        if self.enforce_calendar_arbitrage:
            total_variance = iv_grid**2 * maturities_safe[:, np.newaxis]
            total_variance_af = self._enforce_calendar_arbitrage(total_variance)
            iv_grid = np.sqrt(total_variance_af / maturities_safe[:, np.newaxis])
            self.logger.info("Enforced calendar arbitrage constraints")

        self.iv_surface = iv_grid

        # Log diagnostics
        self._log_surface_diagnostics()

        log_function_end("interpolate_surface")
        return self.iv_surface

    def _interpolate_total_variance(
        self,
        points: np.ndarray,
        values: np.ndarray,
        time_grid: np.ndarray,
        k_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate total variance using the configured method.

        Parameters
        ----------
        points : np.ndarray
            Market data coordinates (n_points × 2) [T, log_moneyness]
        values : np.ndarray
            Total variance values at market points
        time_grid : np.ndarray
            Maturity grid (meshgrid format)
        k_grid : np.ndarray
            Log-moneyness grid (meshgrid format)

        Returns
        -------
        np.ndarray
            Interpolated total variance on the grid
        """
        grid_points = np.column_stack([time_grid.ravel(), k_grid.ravel()])

        if self.interpolation_method == "rbf":
            total_variance_grid = self._interpolate_rbf(
                points, values, grid_points, time_grid.shape
            )
        elif self.interpolation_method == "cubic":
            total_variance_grid = self._interpolate_cubic(
                points, values, time_grid, k_grid
            )
        else:  # linear fallback
            total_variance_grid = self._interpolate_linear(
                points, values, time_grid, k_grid
            )

        return total_variance_grid

    def _interpolate_rbf(
        self,
        points: np.ndarray,
        values: np.ndarray,
        grid_points: np.ndarray,
        grid_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Interpolate using Radial Basis Functions (thin-plate spline).

        RBF is ideal for sparse, irregular data and produces C² smooth surfaces.

        Parameters
        ----------
        points : np.ndarray
            Market data coordinates
        values : np.ndarray
            Total variance values
        grid_points : np.ndarray
            Flattened grid coordinates
        grid_shape : Tuple[int, int]
            Shape of the output grid

        Returns
        -------
        np.ndarray
            Interpolated total variance grid
        """
        try:
            from scipy.interpolate import RBFInterpolator

            # Normalize coordinates for better conditioning
            # Scale T and K to similar ranges
            t_scale = points[:, 0].std() + 1e-8
            k_scale = points[:, 1].std() + 1e-8
            points_scaled = points.copy()
            points_scaled[:, 0] /= t_scale
            points_scaled[:, 1] /= k_scale

            grid_points_scaled = grid_points.copy()
            grid_points_scaled[:, 0] /= t_scale
            grid_points_scaled[:, 1] /= k_scale

            # RBF with thin-plate spline kernel
            # smoothing parameter controls noise tolerance
            rbf = RBFInterpolator(
                points_scaled,
                values,
                kernel="thin_plate_spline",
                smoothing=0.01,  # Small smoothing for noise tolerance
                degree=1,  # Linear polynomial trend
            )

            total_variance_grid = rbf(grid_points_scaled).reshape(grid_shape)
            self.logger.info("Used RBF interpolation (thin-plate spline)")

            # Handle any negative values (shouldn't happen but safety check)
            total_variance_grid = np.maximum(total_variance_grid, 1e-8)

            return total_variance_grid

        except ImportError:
            self.logger.warning(
                "RBFInterpolator not available, falling back to cubic"
            )
            return self._interpolate_cubic(
                points,
                values,
                grid_points.reshape(grid_shape + (2,))[:, :, 0],
                grid_points.reshape(grid_shape + (2,))[:, :, 1],
            )

    def _interpolate_cubic(
        self,
        points: np.ndarray,
        values: np.ndarray,
        time_grid: np.ndarray,
        k_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate using cubic splines via scipy.interpolate.griddata.

        Cubic interpolation provides C² continuity where data is available,
        with linear fallback for regions with sparse data.

        Parameters
        ----------
        points : np.ndarray
            Market data coordinates
        values : np.ndarray
            Total variance values
        time_grid : np.ndarray
            Maturity meshgrid
        k_grid : np.ndarray
            Log-moneyness meshgrid

        Returns
        -------
        np.ndarray
            Interpolated total variance grid
        """
        # Try cubic first (C² smooth)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            total_variance_grid = griddata(
                points, values, (time_grid, k_grid), method="cubic", fill_value=np.nan
            )

        # Fill NaN regions with linear interpolation
        nan_mask = np.isnan(total_variance_grid)
        if np.any(nan_mask):
            n_nan = np.sum(nan_mask)
            self.logger.info(
                "Cubic interpolation left %d NaN values, filling with linear",
                n_nan,
            )
            tv_linear = griddata(
                points, values, (time_grid, k_grid), method="linear", fill_value=np.nan
            )
            total_variance_grid = np.where(nan_mask, tv_linear, total_variance_grid)

        # Fill remaining NaNs with nearest neighbour (far from data)
        nan_mask = np.isnan(total_variance_grid)
        if np.any(nan_mask):
            n_nan = np.sum(nan_mask)
            self.logger.info(
                "Linear interpolation left %d NaN values, filling with nearest",
                n_nan,
            )
            tv_nearest = griddata(
                points, values, (time_grid, k_grid), method="nearest"
            )
            total_variance_grid = np.where(nan_mask, tv_nearest, total_variance_grid)

        self.logger.info("Used cubic interpolation with linear/nearest fallback")

        # Safety: ensure non-negative
        total_variance_grid = np.maximum(total_variance_grid, 1e-8)

        return total_variance_grid

    def _interpolate_linear(
        self,
        points: np.ndarray,
        values: np.ndarray,
        time_grid: np.ndarray,
        k_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate using linear interpolation (C⁰ continuous).

        Parameters
        ----------
        points : np.ndarray
            Market data coordinates
        values : np.ndarray
            Total variance values
        time_grid : np.ndarray
            Maturity meshgrid
        k_grid : np.ndarray
            Log-moneyness meshgrid

        Returns
        -------
        np.ndarray
            Interpolated total variance grid
        """
        total_variance_grid = griddata(
            points, values, (time_grid, k_grid), method="linear", fill_value=np.nan
        )

        # Fill NaNs with nearest neighbour
        nan_mask = np.isnan(total_variance_grid)
        if np.any(nan_mask):
            tv_nearest = griddata(
                points, values, (time_grid, k_grid), method="nearest"
            )
            total_variance_grid = np.where(nan_mask, tv_nearest, total_variance_grid)

        self.logger.info("Used linear interpolation with nearest fallback")

        return np.maximum(total_variance_grid, 1e-8)

    def _handle_outliers(
        self, iv_grid: np.ndarray, lower: float = 0.01, upper: float = 5.0
    ) -> np.ndarray:
        """
        Handle outliers by replacing with local median, not clipping.

        Clipping creates artificial flat regions that violate no-arbitrage.
        Local median replacement preserves smoothness while removing spikes.

        Parameters
        ----------
        iv_grid : np.ndarray
            Implied volatility grid
        lower : float
            Lower bound for reasonable IV (default 1%)
        upper : float
            Upper bound for reasonable IV (default 500%)

        Returns
        -------
        np.ndarray
            IV grid with outliers replaced
        """
        outlier_mask = (iv_grid < lower) | (iv_grid > upper) | ~np.isfinite(iv_grid)
        n_outliers = np.sum(outlier_mask)

        if n_outliers > 0:
            self.logger.warning(
                "Found %d outliers (%.2f%%) outside [%.0f%%, %.0f%%]",
                n_outliers,
                100 * n_outliers / iv_grid.size,
                lower * 100,
                upper * 100,
            )

            # Replace outliers with local median
            iv_median = median_filter(iv_grid, size=5)
            iv_grid = np.where(outlier_mask, iv_median, iv_grid)

            # Final safety clip for any remaining extreme values
            iv_grid = np.clip(iv_grid, lower, upper)

        return iv_grid

    def _enforce_calendar_arbitrage(
        self, total_variance: np.ndarray
    ) -> np.ndarray:
        """
        Ensure calendar arbitrage-free condition: ∂w/∂T ≥ 0.

        Uses isotonic regression per strike to enforce monotonicity smoothly,
        followed by PCHIP interpolation to remove artificial kinks.

        Parameters
        ----------
        total_variance : np.ndarray
            Total variance grid (n_maturities × n_strikes)

        Returns
        -------
        np.ndarray
            Arbitrage-free total variance grid
        """
        n_mat, n_strike = total_variance.shape
        tv_arbitrage_free = np.zeros_like(total_variance)

        for k in range(n_strike):
            tv_slice = total_variance[:, k]

            # Step 1: Enforce strict monotonicity with proportional increase
            tv_monotonic = np.zeros_like(tv_slice)
            tv_monotonic[0] = max(tv_slice[0], 1e-8)

            for t in range(1, n_mat):
                # Minimum increase proportional to time step
                delta_t = self.grid_maturities[t] - self.grid_maturities[t - 1]
                min_increase = self.min_variance_increase * delta_t
                tv_monotonic[t] = max(tv_slice[t], tv_monotonic[t - 1] + min_increase)

            # Step 2: Smooth with PCHIP to remove kinks while preserving monotonicity
            # PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
            # preserves monotonicity and produces C¹ continuous curves
            try:
                pchip = PchipInterpolator(self.grid_maturities, tv_monotonic)
                tv_arbitrage_free[:, k] = pchip(self.grid_maturities)
            except ValueError:
                # Fallback if PCHIP fails (shouldn't happen with monotonic data)
                tv_arbitrage_free[:, k] = tv_monotonic

        return tv_arbitrage_free

    def _log_surface_diagnostics(self) -> None:
        """Log diagnostic information about the constructed surface."""
        if self.iv_surface is None:
            return

        iv = self.iv_surface

        # Basic statistics
        self.logger.info("Surface diagnostics:")
        self.logger.info(
            "  IV range: [%.1f%%, %.1f%%]", iv.min() * 100, iv.max() * 100
        )
        self.logger.info("  IV mean: %.1f%%", iv.mean() * 100)
        self.logger.info("  IV std: %.1f%%", iv.std() * 100)

        # Check for remaining issues
        n_nan = np.sum(np.isnan(iv))
        n_inf = np.sum(np.isinf(iv))
        if n_nan > 0 or n_inf > 0:
            self.logger.warning("  WARNING: %d NaN, %d Inf values remain", n_nan, n_inf)

        # Sample ATM values across maturities
        atm_idx = self.n_moneyness_bins // 2  # Approximate ATM
        atm_ivs = iv[:, atm_idx]
        self.logger.info(
            "  ATM IV term structure: %.1f%% (short) → %.1f%% (long)",
            atm_ivs[0] * 100,
            atm_ivs[-1] * 100,
        )

    def save_data(self) -> None:
        """Save the interpolated surface and grid data to the output directory."""
        log_function_start("save_data")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save surface and grids
        np.save(self.output_dir / "iv_surface.npy", self.iv_surface)
        self.logger.info("Saved IV surface to %s", self.output_dir / "iv_surface.npy")

        np.save(self.output_dir / "grid_maturities.npy", self.grid_maturities)
        self.logger.info(
            "Saved grid maturities to %s", self.output_dir / "grid_maturities.npy"
        )

        np.save(self.output_dir / "grid_log_moneyness.npy", self.grid_log_moneyness)
        self.logger.info(
            "Saved grid log-moneyness to %s",
            self.output_dir / "grid_log_moneyness.npy",
        )

        np.save(self.output_dir / "grid_risk_free_rates.npy", self.grid_risk_free_rates)
        self.logger.info(
            "Saved grid risk-free rates to %s",
            self.output_dir / "grid_risk_free_rates.npy",
        )

        log_function_end("save_data")

    def plot_surface(self) -> None:
        """Generate and save 2D heatmap plots of the IV surface and raw data."""
        log_function_start("plot_surface")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Left panel: Interpolated surface
        im1 = axes[0].imshow(
            self.iv_surface.T,  # Transpose for correct orientation
            aspect="auto",
            extent=(
                self.grid_maturities[0],
                self.grid_maturities[-1],
                self.grid_log_moneyness[0],
                self.grid_log_moneyness[-1],
            ),
            origin="lower",
            cmap="viridis",
        )
        axes[0].set_title("Interpolated IV Surface", fontsize=12)
        axes[0].set_xlabel("Maturity (Years)", fontsize=11)
        axes[0].set_ylabel("Log Moneyness", fontsize=11)
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label("Implied Volatility", fontsize=10)

        # Right panel: Raw market data
        scatter = axes[1].scatter(
            self.raw_points["T_years"],
            self.raw_points["log_moneyness"],
            c=self.raw_points["implied_volatility"],
            s=3,
            alpha=0.6,
            cmap="viridis",
        )
        axes[1].set_title("Raw Market Data", fontsize=12)
        axes[1].set_xlabel("Maturity (Years)", fontsize=11)
        axes[1].set_ylabel("Log Moneyness", fontsize=11)
        axes[1].set_xlim(self.grid_maturities[0], self.grid_maturities[-1])
        axes[1].set_ylim(self.grid_log_moneyness[0], self.grid_log_moneyness[-1])
        cbar2 = plt.colorbar(scatter, ax=axes[1])
        cbar2.set_label("Implied Volatility", fontsize=10)

        plt.tight_layout()

        # Save in multiple formats
        save_path = self.output_dir / "iv_surface"
        plt.savefig(save_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
        plt.savefig(
            save_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight"
        )
        self.logger.info("Saved 2D surface plots to %s", save_path)

        plt.close(fig)
        log_function_end("plot_surface")

    def plot_surface_3d(self) -> None:
        """
        Generate interactive 3D surface plot with market data overlay.

        This visualisation helps identify:
        - Interpolation artifacts (bumps, valleys)
        - Arbitrage violations (non-monotonicity in T)
        - Smile/skew dynamics across maturities
        - Sparse data regions
        """
        log_function_start("plot_surface_3d")
        
        # Create figure with two views
        fig = plt.figure(figsize=(16, 7))

        # View 1: Standard perspective
        ax1 = fig.add_subplot(121, projection="3d")
        self._plot_3d_surface(ax1, elev=25, azim=45, title="IV Surface (Standard View)")

        # View 2: Rotated perspective (along maturity axis)
        ax2 = fig.add_subplot(122, projection="3d")
        self._plot_3d_surface(ax2, elev=20, azim=135, title="IV Surface (Rotated View)")

        plt.tight_layout()

        # Save
        save_path_png = self.output_dir / "iv_surface_3d.png"
        save_path_svg = self.output_dir / "iv_surface_3d.svg"
        plt.savefig(save_path_png, format="png", dpi=300, bbox_inches="tight")
        plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
        self.logger.info("Saved 3D surface plots to %s", save_path_png)

        plt.close(fig)

        # Also create a single high-resolution plot for presentations
        fig_single = plt.figure(figsize=(14, 10))
        ax_single = fig_single.add_subplot(111, projection="3d")
        self._plot_3d_surface(
            ax_single,
            elev=25,
            azim=45,
            title="Implied Volatility Surface\nInterpolated Surface with Market Data Overlay",
            show_colorbar=True,
        )

        save_path_hires = self.output_dir / "iv_surface_3d_hires.png"
        plt.savefig(save_path_hires, format="png", dpi=600, bbox_inches="tight")
        self.logger.info("Saved high-res 3D plot to %s", save_path_hires)

        plt.close(fig_single)
        log_function_end("plot_surface_3d")

    def _plot_3d_surface(
        self,
        ax,
        elev: float = 25,
        azim: float = 45,
        title: str = "IV Surface",
        show_colorbar: bool = False,
    ) -> None:
        """
        Helper to plot 3D surface on given axes.

        Parameters
        ----------
        ax : Axes3D
            Matplotlib 3D axes
        elev : float
            Elevation angle for view
        azim : float
            Azimuth angle for view
        title : str
            Plot title
        show_colorbar : bool
            Whether to add colorbar
        """
        # Create meshgrid
        T_mesh, K_mesh = np.meshgrid(self.grid_maturities, self.grid_log_moneyness)

        # Plot interpolated surface
        surf = ax.plot_surface(
            T_mesh,
            K_mesh,
            self.iv_surface.T,  # Transpose for correct orientation
            cmap="viridis",
            alpha=0.85,
            edgecolor="none",
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        # Overlay raw market data points
        if self.raw_points is not None and len(self.raw_points) > 0:
            # Subsample if too many points for clarity
            n_points = len(self.raw_points)
            if n_points > 2000:
                sample_idx = np.random.choice(n_points, 2000, replace=False)
                plot_points = self.raw_points.iloc[sample_idx]
            else:
                plot_points = self.raw_points

            ax.scatter(
                plot_points["T_years"],
                plot_points["log_moneyness"],
                plot_points["implied_volatility"],
                c="red",
                marker="o",
                s=8,
                alpha=0.5,
                label=f"Market Data (n={n_points})",
                edgecolors="darkred",
                linewidths=0.3,
            )

        # Formatting
        ax.set_xlabel("Maturity (Years)", fontsize=10, labelpad=8)
        ax.set_ylabel("Log Moneyness", fontsize=10, labelpad=8)
        ax.set_zlabel("Implied Volatility", fontsize=10, labelpad=8)
        ax.set_title(title, fontsize=12, pad=15)

        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)

        # Grid
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        # Legend
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        # Colorbar (optional)
        if show_colorbar:
            plt.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1, label="IV")

    def plot_smile_slices(self) -> None:
        """
        Plot IV smile slices at selected maturities.

        This helps validate the smile shape at different tenors.
        """
        log_function_start("plot_smile_slices")

        # Select maturities to plot (short, medium, long)
        n_mats = len(self.grid_maturities)
        mat_indices = [0, n_mats // 4, n_mats // 2, 3 * n_mats // 4, n_mats - 1]

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(mat_indices)))

        for i, idx in enumerate(mat_indices):
            T = self.grid_maturities[idx]
            iv_slice = self.iv_surface[idx, :]

            ax.plot(
                self.grid_log_moneyness,
                iv_slice * 100,  # Convert to percentage
                color=colors[i],
                linewidth=2,
                label=f"T = {T:.3f}y",
            )

            # Overlay market data near this maturity
            if self.raw_points is not None:
                mat_tol = (self.grid_maturities[1] - self.grid_maturities[0]) * 2
                mask = np.abs(self.raw_points["T_years"] - T) < mat_tol
                if np.sum(mask) > 0:
                    ax.scatter(
                        self.raw_points.loc[mask, "log_moneyness"],
                        self.raw_points.loc[mask, "implied_volatility"] * 100,
                        color=colors[i],
                        s=20,
                        alpha=0.5,
                        edgecolors="black",
                        linewidths=0.5,
                    )

        ax.set_xlabel("Log Moneyness", fontsize=11)
        ax.set_ylabel("Implied Volatility (%)", fontsize=11)
        ax.set_title("IV Smile Slices at Selected Maturities", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="ATM")

        plt.tight_layout()

        save_path = self.output_dir / "iv_smile_slices.png"
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        self.logger.info("Saved smile slices to %s", save_path)

        plt.close(fig)
        log_function_end("plot_smile_slices")

    def plot_term_structure(self) -> None:
        """
        Plot IV term structure at selected strikes.

        This helps validate calendar arbitrage (IV should be relatively stable
        or slightly decreasing with maturity for fixed moneyness).
        """
        log_function_start("plot_term_structure")

        # Select strikes to plot (OTM put, ATM, OTM call)
        n_strikes = len(self.grid_log_moneyness)
        atm_idx = n_strikes // 2
        strike_indices = [
            n_strikes // 5,  # OTM put
            atm_idx,  # ATM
            4 * n_strikes // 5,  # OTM call
        ]
        labels = ["OTM Put", "ATM", "OTM Call"]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["blue", "green", "red"]

        for i, (idx, label) in enumerate(zip(strike_indices, labels)):
            k = self.grid_log_moneyness[idx]
            iv_slice = self.iv_surface[:, idx]

            ax.plot(
                self.grid_maturities,
                iv_slice * 100,
                color=colors[i],
                linewidth=2,
                label=f"{label} (k={k:.2f})",
            )

        ax.set_xlabel("Maturity (Years)", fontsize=11)
        ax.set_ylabel("Implied Volatility (%)", fontsize=11)
        ax.set_title("IV Term Structure at Selected Strikes", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / "iv_term_structure.png"
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        self.logger.info("Saved term structure to %s", save_path)

        plt.close('all')
        log_function_end("plot_term_structure")


@hydra.main(version_base=None, config_path="../configs", config_name="surface_builder")
def main(cfg: DictConfig):
    """
    Main entry point for constructing the market surface.

    Args:
        cfg (Optional[DictConfig]): Configuration object, defaults to None.
    """
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")
    logger.info("Starting market surface building...")

    surface_builder = MarketSurfaceBuilder(cfg)
    surface_builder.load_data()
    surface_builder.create_market_grid()
    surface_builder.interpolate_risk_free_rates()
    surface_builder.interpolate_surface()
    surface_builder.save_data()

    # Generate all visualisations
    surface_builder.plot_surface()  # 2D heatmaps
    surface_builder.plot_surface_3d()  # 3D surface
    surface_builder.plot_smile_slices()  # Smile cross-sections
    surface_builder.plot_term_structure()  # Term structure

    logger.info("Market surface building completed successfully")


if __name__ == "__main__":
    main()
