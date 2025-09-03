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
        df.drop(['secid', 'optionid'], axis=1, inplace=True)
        self.logger.info("Dropped the following columns: ['secid', 'optionid']")

        min_iv = self.cfg.data_processor.min_iv
        max_iv = self.cfg.data_processor.max_iv
        moneyness_min = self.cfg.data_processor.moneyness_min
        moneyness_max = self.cfg.data_processor.moneyness_max
        min_open_interest = self.cfg.data_processor.min_open_interest
        r = self.cfg.data_processor.risk_free_rate
        q = self.cfg.data_processor.dividend_yield
        days_in_year = self.cfg.data_processor.days_in_year

        numeric_cols = [c for c in ['best_bid',
                                    'best_offer',
                                    'mid_quote',
                                    'volume',
                                    'implied_volatility',
                                    'strike_price',
                                    'open_interest',
                                    'underlying_close',
                                    'time_to_expiration'] if c in df.columns]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        essential = ['strike_price', 'underlying_close', 'expiration']
        for c in essential:
            if c not in df.columns:
                raise KeyError(f"Required column '{c}' not found in input data")

        if 'best_bid' in df.columns and 'best_offer' in df.columns:
            df = df[(df['best_bid'] > 0) & (df['best_offer'] > 0)]

        if 'volume' in df.columns:
            df = df[df['volume'] > 0]

        if 'implied_volatility' in df.columns:
            iv = df['implied_volatility'].dropna()

            if not iv.empty and iv.max() > 10:
                df['implied_volatility'] = df['implied_volatility'] / 100.0

            df = df[(df['implied_volatility'] >= min_iv) & (df['implied_volatility'] <= max_iv)]


        if 'best_bid' in df.columns and 'best_offer' in df.columns:
            df['mid'] = (df['best_bid'] + df['best_offer']) / 2.0
        elif 'mid_quote' in df.columns:
            df['mid'] = df['mid_quote']
        else:
            df['mid'] = np.nan
        self.logger.info("Calculated 'mid' prices")


        if 'time_to_expiration' in df.columns and df['time_to_expiration'].notna().any():

            df['T_years'] = df['time_to_expiration'].astype(float) / days_in_year
        else:

            if 'date' in df.columns and 'expiration' in df.columns:
                try:
                    dates = pd.to_datetime(df['date'])
                    exps = pd.to_datetime(df['expiration'])
                    df['T_years'] = (exps - dates).dt.days.astype(float) / days_in_year
                except (ValueError, TypeError, OverflowError, pd.errors.OutOfBoundsDatetime):
                    df['T_years'] = 0.0
            else:
                df['T_years'] = 0.0
        self.logger.info("Calculated 'T_years'")


        df['T_years'] = df['T_years'].clip(lower=1e-6)
        self.logger.info("Clipped 'T_years' to avoid zero or negative values")


        underlying = df['underlying_close'].astype(float)
        strike = df['strike_price'].astype(float)
        df['moneyness'] = strike / underlying
        df = df[(df['moneyness'] >= moneyness_min) & (df['moneyness'] <= moneyness_max)]
        self.logger.info("Filtered data based on moneyness between %.2f and %.2f", moneyness_min, moneyness_max)

        df['forward'] = underlying * np.exp((r - q) * df['T_years'])
        self.logger.info("Calculated 'forward' prices")

        df['log_moneyness'] = np.log(strike / df['forward'].clip(lower=1e-12))
        self.logger.info("Calculated 'log_moneyness'")

        if 'open_interest' in df.columns:
            oi_by_exp = df.groupby('expiration')['open_interest'].sum()
            valid_exps = oi_by_exp[oi_by_exp >= min_open_interest].index
            df = df[df['expiration'].isin(valid_exps)]

        df.dropna(subset=['mid', 'moneyness', 'T_years', 'log_moneyness'], inplace=True)
        self.logger.info("Processed data now has %d rows and %d columns", df.shape[0], df.shape[1])

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
