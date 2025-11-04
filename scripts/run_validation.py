"""
Validation runner for TSLA option data processing.

This module orchestrates the loading and processing of TSLA option data using
configuration-driven data loading and processing classes. It integrates with Hydra
for configuration management, sets up logging, and loads environment variables.

Attributes:
    None
"""
from typing import Optional
from omegaconf import DictConfig
from dotenv import load_dotenv
import hydra

from scripts.logging_config import setup_logging, get_logger
from src.data_loader import DataLoader
from src.data_processor import DataProcessor

load_dotenv()

class ValidationRunner:
    """
    Runner class for validating TSLA option data loading and processing.

    This class coordinates the execution of data loading and processing tasks,
    using instances of DataLoader and DataProcessor, with configuration provided
    via Hydra.

    Attributes:
        cfg (DictConfig): Configuration object containing runtime parameters.
        logger (logging.Logger): Logger instance for tracking operations.
        data_loader (DataLoader): Instance for loading data from the database.
        data_processor (DataProcessor): Instance for processing loaded data.
    """
    def __init__(self, cfg: DictConfig, data_loader: DataLoader, data_processor: DataProcessor):
        self.cfg = cfg
        self.logger = get_logger("ValidationRunner")
        self.data_loader = data_loader
        self.data_processor = data_processor

    def run(self):
        """
        Execute the data loading and processing pipeline.

        Connects to the database, loads data, saves it, processes the data, and
        saves the processed results. Logs progress and handles errors appropriately.

        Raises:
            Exception: If an error occurs during data loading or processing.
        """
        get_data = self.data_loader
        try:
            get_data.connect_to_database()
            df = get_data.load_data()
            get_data.save_data(df)

            self.logger.info("Data loading completed successfully.")
        except Exception as e:
            self.logger.error("Error during data loading: {%s}", e)
            raise
        # Use the DataProcessor instance's methods (process_data, save_data)
        try:
            processed_df = self.data_processor.process_data()
            self.data_processor.save_data(processed_df)
            self.logger.info("Data processing completed successfully.")
        except Exception as e:
            self.logger.error("Error during data processing: {%s}", e)
            raise

        print("Validation completed.")


@hydra.main(config_path="../configs", config_name="run_validation")
def main(cfg: Optional[DictConfig] = None):
    """
    Main entry point for the validation process.

    Initialises logging, creates instances of DataLoader and DataProcessor,
    and runs the ValidationRunner to execute the data pipeline.

    Args:
        cfg (Optional[DictConfig], optional): Configuration object for the pipeline.
            Defaults to None, in which case Hydra loads the configuration.
    """
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("run_validation")
    logger.info("Starting validation process")

    data_loader = DataLoader(cfg)
    data_processor = DataProcessor(cfg)
    validation_runner = ValidationRunner(cfg,
                                            data_loader,
                                            data_processor,
                                            )
    validation_runner.run()

if __name__ == "__main__":
    main()
