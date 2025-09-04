"""
Market surface builder module.

This module provides functionality to construct and visualise implied volatility surfaces
from market data using interpolation techniques.
"""
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig
import hydra
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end

class MarketSurfaceBuilder:
    """
    A class to build and visualise implied volatility surfaces from market data.

    Attributes:
        cfg (DictConfig): Configuration object containing paths and parameters.
        logger (logging.Logger): Logger instance for logging operations.
        input_path (Path): Path to the input market data file.
        output_dir (Path): Directory to save output files.
        df (pd.DataFrame): DataFrame holding the loaded market data.
        n_maturity_bins (int): Number of bins for maturity grid.
        n_moneyness_bins (int): Number of bins for moneyness grid.
        grid_maturities (np.ndarray): Array of maturity grid points.
        grid_log_moneyness (np.ndarray): Array of log moneyness grid points.
        raw_points (pd.DataFrame): DataFrame containing raw data points for interpolation.
        iv_surface (np.ndarray): Implied volatility surface grid.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.logger = get_logger("MarketSurfaceBuilder")
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        input_path = self.cfg.data_processor.output_path
        self.input_path = Path(repo_root / input_path).resolve()
        self.output_dir = Path(repo_root / self.cfg.surface_builder.output_dir).resolve()
        self.df = None

        self.n_maturity_bins = self.cfg.surface_builder.n_maturity_bins
        self.n_moneyness_bins = self.cfg.surface_builder.n_moneyness_bins
        self.grid_maturities = None
        self.grid_log_moneyness = None

        self.raw_points = None
        self.iv_surface = None

    def load_data(self):
        """
        Load market data from the input path.
        """
        log_function_start("load_data")
        self.df = pd.read_csv(self.input_path)
        self.logger.info("Loaded market data from %s", self.input_path)
        log_function_end("load_data")

    def create_market_grid(self):
        """
        Create a grid of maturities and log moneyness for interpolation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of maturity and log moneyness grid points.
        """
        log_function_start("create_market_grid")

        time_min = self.df["T_years"].min()
        time_max = self.df["T_years"].max()
        self.logger.info("Maturity range: [%f, %f]", time_min, time_max)
        k_min = self.df["log_moneyness"].min()
        k_max = self.df["log_moneyness"].max()
        self.logger.info("Moneyness range: [%f, %f]", k_min, k_max)

        self.grid_maturities = np.linspace(time_min, time_max, self.n_maturity_bins)
        self.grid_log_moneyness = np.linspace(k_min, k_max, self.n_moneyness_bins)
        self.logger.info("Created market grid with %d maturities and %d moneyness levels",
                            self.n_maturity_bins, self.n_moneyness_bins)

        log_function_end("create_market_grid")
        return self.grid_maturities, self.grid_log_moneyness

    def interpolate_surface(self):
        """
        Interpolate the implied volatility surface using market data.

        Returns:
            np.ndarray: Interpolated implied volatility surface.
        """
        log_function_start("interpolate_surface")

        self.df['valid_pricing'] = self.df['valid_pricing'].astype(bool)
        valid_df = self.df[self.df['valid_pricing']].copy()
        self.logger.info("Number of valid data points: %d", len(valid_df))
        self.raw_points = valid_df[["T_years", "log_moneyness", "implied_volatility"]].copy()
        self.logger.info("Extracted raw points for interpolation")

        valid_df["total_variance"] = valid_df["implied_volatility"] ** 2 * valid_df["T_years"]
        self.logger.info("Computed total variance for valid data points")
        points = valid_df[["T_years", "log_moneyness"]].values
        values = valid_df["total_variance"].values
        self.logger.info("Prepared points and values for interpolation")

        time_grid, k_grid = np.meshgrid(self.grid_maturities,
                                        self.grid_log_moneyness,
                                        indexing='ij',
                                        )
        self.logger.info("Created meshgrid for interpolation")
        self.logger.info("Starting griddata interpolation...")
        total_variance_grid = griddata(points,
                                        values,
                                        (time_grid, k_grid),
                                        method='nearest',
                                        fill_value=np.nan
                                        )
        total_variance_grid = gaussian_filter(total_variance_grid, sigma=0.5, mode='nearest')
        self.logger.info("Performed griddata interpolation for total variance")

        for col_idx in range(total_variance_grid.shape[1]):
            var_slice = total_variance_grid[:, col_idx].copy()
            valid_mask = ~np.isnan(var_slice)
            if valid_mask.sum() > 1:
                for row_idx in range(1, len(var_slice)):
                    if (valid_mask[row_idx] and valid_mask[row_idx - 1]
                            and var_slice[row_idx] < var_slice[row_idx - 1]):
                        var_slice[row_idx] = var_slice[row_idx - 1]
            total_variance_grid[:, col_idx] = var_slice

        self.iv_surface = np.sqrt(total_variance_grid / time_grid)
        self.logger.info("Interpolated implied volatility surface on the market grid")

        log_function_end("interpolate_surface")
        return self.iv_surface

    def save_data(self):
        """
        Save the interpolated surface and grid data to the output directory.
        """
        log_function_start("save_data")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.output_dir / "iv_surface.npy", self.iv_surface)
        self.logger.info("Saved implied volatility surface to %s",
                            self.output_dir / "iv_surface.npy")
        np.save(self.output_dir / "grid_maturities.npy", self.grid_maturities)
        self.logger.info("Saved grid maturities to %s", self.output_dir / "grid_maturities.npy")
        np.save(self.output_dir / "grid_log_moneyness.npy", self.grid_log_moneyness)
        self.logger.info("Saved grid log moneyness to %s",
                            self.output_dir / "grid_log_moneyness.npy")

        log_function_end("save_data")

    def plot_surface(self):
        """
        Generate and save a plot of the implied volatility surface and raw data.
        """
        log_function_start("plot_surface")

        fig, axes = plt.subplots(1,2, figsize=(15,6))
        _unused = fig
        im1 = axes[0].imshow(self.iv_surface.maturity,
                                aspect='auto',
                                extent=(self.grid_maturities[0], self.grid_maturities[-1],
                                        self.grid_log_moneyness[0], self.grid_log_moneyness[-1]),
                                )

        axes[0].set_title("Implied Volatility Surface")
        axes[0].set_xlabel("Maturity (Years)")
        axes[0].set_ylabel("Log Moneyness")
        plt.colorbar(im1, ax=axes[0])

        scatter = axes[1].scatter(self.raw_points["T_years"],
                                        self.raw_points["log_moneyness"],
                                        c=self.raw_points["implied_volatility"],
                                        s=1,
                                        alpha=0.5)
        axes[1].set_title("Raw Option Data")
        axes[1].set_xlabel("Maturity (Years)")
        axes[1].set_ylabel("Log Moneyness")
        plt.colorbar(scatter, ax=axes[1])

        plt.tight_layout()
        save_path = Path(self.output_dir / "iv_surface.png")
        plt.savefig(save_path)
        self.logger.info("Saved plots to %s", save_path)

        log_function_end("plot_surface")

@hydra.main(version_base=None, config_path="../configs", config_name="surface_builder")
def main(cfg: Optional[DictConfig] = None):
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
    surface_builder.interpolate_surface()
    surface_builder.save_data()
    surface_builder.plot_surface()

if __name__ == "__main__":
    main()
