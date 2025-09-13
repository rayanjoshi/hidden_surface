"""
Data processing module for financial data analysis.

This module provides functionality to process financial datasets, applying filters and
calculations based on configuration parameters. It uses pandas for data manipulation
and hydra for configuration management.
"""
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import DictConfig
import hydra

from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end

class DataProcessor:
    """
    Processes financial data according to specified configuration parameters.

    Attributes:
        cfg (DictConfig): Configuration object containing data processing parameters.
        input_path (Path): Path to the input data file.
        output_path (Path): Path where processed data will be saved.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.logger = get_logger("DataProcessor")
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        self.input_path = Path(repo_root / self.cfg.data_loader.output_path).resolve()
        self.output_path = Path(repo_root / self.cfg.data_processor.output_path).resolve()

    def process_data(self):
        """
        Process financial data by applying filters and calculating derived columns.

        Reads input data, applies data cleaning, filters based on configuration parameters,
        and calculates financial metrics such as moneyness and forward prices.

        Returns:
            pd.DataFrame: Processed DataFrame with filtered and calculated columns.

        Raises:
            KeyError: If required columns are missing from the input data.
        """
        log_function_start("process_data")
        df = pd.read_csv(self.input_path)
        self.logger.info("Loaded data with %d rows and %d columns", df.shape[0], df.shape[1])
        df.dropna(inplace=True)
        self.logger.info("Dropped rows with NaN values; remaining rows: %d", len(df))
        df.drop(['secid', 'optionid'], axis=1, inplace=True, errors='ignore')
        self.logger.info("Dropped unnecessary columns")

        # Config params (add these to your config.yaml)
        min_iv = self.cfg.data_processor.min_iv
        max_iv = self.cfg.data_processor.get('max_iv', 2.0)
        moneyness_min = self.cfg.data_processor.moneyness_min
        moneyness_max = self.cfg.data_processor.moneyness_max
        min_open_interest = self.cfg.data_processor.get('min_open_interest', 10)
        min_ttm = self.cfg.data_processor.get('min_ttm', 7 / 365.0)
        max_bid_ask_spread = self.cfg.data_processor.get('max_bid_ask_spread', 0.2)
        iv_outlier_z_threshold = self.cfg.data_processor.get('iv_outlier_z_threshold', 2.5)
        r = df['risk_free_rate']
        q = self.cfg.data_processor.dividend_yield
        days_in_year = self.cfg.data_processor.days_in_year

        numeric_cols = [c for c in ['best_bid', 'best_offer', 'mid_quote',
                                    'volume', 'implied_volatility',
                                    'strike_price', 'open_interest',
                                    'underlying_close', 'time_to_expiration',
                                    ]
                        if c in df.columns]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        essential = ['strike_price', 'underlying_close', 'expiration', 'cp_flag', 'date', 'risk_free_rate']
        for c in essential:
            if c not in df.columns:
                raise KeyError(f"Required column '{c}' not found in input data")

        df['date'] = pd.to_datetime(df['date'])
        df['expiration'] = pd.to_datetime(df['expiration'])

        # Liquidity filters
        if 'best_bid' in df.columns and 'best_offer' in df.columns:
            df = df[(df['best_bid'] > 0) & (df['best_offer'] > 0)]
            df['bid_ask_spread'] = (
                (df['best_offer'] - df['best_bid']) /
                ((df['best_bid'] + df['best_offer']) / 2)
            )
            df = df[df['bid_ask_spread'] <= max_bid_ask_spread]
            removed_rows = len(df) - len(df[df['bid_ask_spread'] <= max_bid_ask_spread])
            self.logger.info(
                "Filtered on relative bid-ask spread <= %.2f (removed %d rows)",
                max_bid_ask_spread, removed_rows
            )

        if 'volume' in df.columns:
            df = df[df['volume'] > 0]

        if 'open_interest' in df.columns:
            df = df[df['open_interest'] >= min_open_interest]
            self.logger.info("Filtered on per-option open interest >= %d", min_open_interest)

        # IV filtering (no normalization needed; data is in decimals)
        if 'implied_volatility' in df.columns:
            df = df[(df['implied_volatility'] >= min_iv) & (df['implied_volatility'] <= max_iv)]
            self.logger.info("Filtered IV between %.2f and %.2f", min_iv, max_iv)

            # Per-date, per-expiration IV outlier removal
            def remove_iv_outliers(group):
                if len(group) < 3:
                    return group
                mean_iv = group['implied_volatility'].mean()
                std_iv = group['implied_volatility'].std()
                if std_iv == 0:
                    return group
                z_scores = (group['implied_volatility'] - mean_iv) / std_iv
                return group[np.abs(z_scores) <= iv_outlier_z_threshold]

            df = df.groupby(['date', 'expiration'], group_keys=False).apply(
                            remove_iv_outliers, include_groups=True
                            ).reset_index(drop=True)
            self.logger.info("Removed IV outliers (z-score threshold: %.1f)",
                                iv_outlier_z_threshold)

        # Calculate mid price
        if 'best_bid' in df.columns and 'best_offer' in df.columns:
            df['mid'] = (df['best_bid'] + df['best_offer']) / 2.0
        elif 'mid_quote' in df.columns:
            df['mid'] = df['mid_quote']
        else:
            df['mid'] = np.nan
        self.logger.info("Calculated 'mid' prices")

        # Calculate T_years
        if 'time_to_expiration' in df.columns and df['time_to_expiration'].notna().any():
            df['T_years'] = df['time_to_expiration'].astype(float)
        else:
            try:
                df['T_years'] = (df['expiration'] - df['date']).dt.days.astype(float) / days_in_year
            except (ValueError, TypeError, OverflowError, pd.errors.OutOfBoundsDatetime):
                df['T_years'] = 0.0
        df['T_years'] = df['T_years'].clip(lower=1e-6)
        df = df[df['T_years'] >= min_ttm]
        self.logger.info("Filtered T_years >= %.4f (min_ttm)", min_ttm)

        # Moneyness filter
        underlying = df['underlying_close'].astype(float)
        strike = df['strike_price'].astype(float)
        df['moneyness'] = strike / underlying
        df = df[(df['moneyness'] >= moneyness_min) & (df['moneyness'] <= moneyness_max)]
        self.logger.info(
            "Filtered data based on moneyness between %.2f and %.2f",
            moneyness_min, moneyness_max
        )

        # Forward and log moneyness
        df['forward'] = underlying * np.exp((r - q) * df['T_years'])
        self.logger.info("Calculated 'forward' prices")
        df['log_moneyness'] = np.log(strike / df['forward'].clip(lower=1e-12))
        self.logger.info("Calculated 'log_moneyness'")

        # No-arbitrage price bounds for calls
        if 'cp_flag' in df.columns and 'mid' in df.columns:
            # Reassign variables after filtering to match the current DataFrame
            underlying = df['underlying_close'].astype(float)
            strike = df['strike_price'].astype(float)
            r = df['risk_free_rate'].astype(float)
            exp_qt = np.exp(-q * df['T_years'])
            exp_rt = np.exp(-r * df['T_years'])
            intrinsic = np.where(
                df['cp_flag'] == 'C',
                np.maximum(0, underlying * exp_qt - strike * exp_rt),
                np.maximum(0, strike * exp_rt - underlying * exp_qt)
            )
            pre_count = len(df)
            df = df[df['mid'] >= intrinsic]
            self.logger.info(
                "Applied no-arbitrage bounds (removed %d violations)",
                pre_count - len(df)
            )

        # Convexity check for calls and puts
        if 'cp_flag' in df.columns:
            def check_monotonicity_and_convexity_calls(group):
                group = group.sort_values('strike_price')
                prices = group['mid'].values
                # Monotonicity check: Call prices should be non-increasing
                if not (prices[:-1] >= prices[1:]).all():
                    self.logger.warning(
                        "Monotonicity violation detected in call group: %s",
                        group['expiration'].iloc[0]
                    )
                    # Keep middle 50% of strikes
                    mid_idx = range(len(group)//4, 3*len(group)//4)
                    return group.iloc[mid_idx]
                # Convexity check: Second differences should be non-negative
                if len(prices) >= 3:
                    # Compute first differences: delta_price = price[i+1] - price[i]
                    first_diffs = np.diff(prices)
                    # Compute second differences: (price[i+2] - price[i+1]) - (price[i+1] - price[i])
                    second_diffs = np.diff(first_diffs)
                    if not (second_diffs >= 0).all():
                        self.logger.warning(
                            "Convexity violation detected in call group: %s",
                            group['expiration'].iloc[0]
                        )
                        # Keep middle 50% of strikes
                        mid_idx = range(len(group)//4, 3*len(group)//4)
                        return group.iloc[mid_idx]
                return group

            def check_monotonicity_and_convexity_puts(group):
                if len(group) < 3 or not all(group['cp_flag'] == 'P'):
                    return group
                group = group.sort_values('strike_price')
                prices = group['mid'].values
                # Monotonicity check: Put prices should be non-decreasing
                if not (prices[:-1] <= prices[1:]).all():
                    self.logger.warning(
                        "Monotonicity violation detected in put group: %s",
                        group['expiration'].iloc[0]
                    )
                    # Keep middle 50% of strikes
                    mid_idx = range(len(group)//4, 3*len(group)//4)
                    return group.iloc[mid_idx]
                # Convexity check: Second differences should be non-negative
                if len(prices) >= 3:
                    first_diffs = np.diff(prices)
                    second_diffs = np.diff(first_diffs)
                    if not (second_diffs >= 0).all():
                        self.logger.warning(
                            "Convexity violation detected in put group: %s",
                            group['expiration'].iloc[0]
                        )
                        # Keep middle 50% of strikes
                        mid_idx = range(len(group)//4, 3*len(group)//4)
                        return group.iloc[mid_idx]
                return group

            df_calls = (
                df[df['cp_flag'] == 'C']
                .groupby('expiration', group_keys=False)
                .apply(check_monotonicity_and_convexity_calls, include_groups=True)
                .reset_index(drop=True)
            )
            df_puts = (
                df[df['cp_flag'] == 'P']
                .groupby('expiration', group_keys=False)
                .apply(check_monotonicity_and_convexity_puts, include_groups=True)
                .reset_index(drop=True)
            )
            df = pd.concat([df_calls, df_puts], ignore_index=True)
            self.logger.info("Applied monotonicity and convexity checks for calls and puts")

        # Expiration-level open interest filter
        if 'open_interest' in df.columns:
            oi_by_exp = df.groupby('expiration')['open_interest'].sum()
            valid_exps = oi_by_exp[oi_by_exp >= min_open_interest].index
            df = df[df['expiration'].isin(valid_exps)]
            self.logger.info(
                "Filtered expirations with total open interest >= %d",
                min_open_interest
            )

        df.dropna(subset=['mid', 'moneyness', 'T_years', 'log_moneyness'], inplace=True)
        self.logger.info("Processed data now has %d rows and %d columns", df.shape[0], df.shape[1])

        # Log IV stats per expiration for debugging
        if 'implied_volatility' in df.columns:
            iv_stats = df.groupby('expiration')['implied_volatility'].describe()
            for exp in iv_stats.index:
                stats = iv_stats.loc[exp]
                self.logger.debug(
                    "IV stats for expiration %s: min=%.2f, max=%.2f, mean=%.2f, std=%.2f",
                    exp, stats['min'], stats['max'], stats['mean'], stats['std']
                )

        log_function_end("process_data")
        return df

    def save_data(self, df: pd.DataFrame):
        """
        Save processed DataFrame to the output path.

        Args:
            df: DataFrame to be saved as a CSV file.
        """
        log_function_start("save_data")
        self.logger.info("Saving processed data to %s", self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        self.logger.info("Processed data saved successfully")
        log_function_end("save_data")

@hydra.main(version_base=None, config_path="../configs", config_name="data_processor")
def main(cfg: Optional[DictConfig] = None):
    """
    Main entry point for data processing.

    Initializes DataProcessor with configuration, processes the data,
    and saves the results.

    Args:
        cfg: Configuration object, defaults to None.
    """
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")
    logger.info("Starting data processing...")
    data_processor = DataProcessor(cfg)
    df = data_processor.process_data()
    data_processor.save_data(df)

if __name__ == "__main__":
    main()
