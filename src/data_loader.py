"""
Data loader module for fetching and saving data from WRDS database.

This module provides functionality to connect to the WRDS database, execute SQL
queries, and save the resulting data to a CSV file. It uses Hydra for configuration
management and integrates logging for monitoring the data loading process.

Dependencies:
    - os
    - pathlib
    - dotenv
    - omegaconf
    - hydra
    - wrds
    - scripts.logging_config
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig
import hydra
import wrds
from scripts.logging_config import (
    get_logger,
    setup_logging,
    log_function_start,
    log_function_end,
)

load_dotenv()


class DataLoader:
    """
    A class to handle data loading from WRDS database and saving to CSV.

    Attributes:
        cfg (DictConfig): Hydra configuration object containing data loader settings.
        db (wrds.Connection): Connection to the WRDS database.
        logger (logging.Logger): Logger instance for tracking operations.
        username (str): WRDS database username from environment variables.
        password (str): WRDS database password from environment variables.
        sql_query (str): Path to the SQL query file from configuration.
        sql_query_path (Path): Resolved path to the SQL query file.
        save_path (Path): Resolved path where the data will be saved as CSV.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.logger = get_logger("DataLoader")
        self.username = os.getenv("WRDS_USERNAME")
        self.password = os.getenv("WRDS_PASSWORD")
        self.sql_query = self.cfg.data_loader.sql_query
        src_dir = Path(__file__).parent
        repo_root = src_dir.parent
        self.sql_query_path = Path(repo_root / self.sql_query).absolute()
        self.save_path = self.cfg.data_loader.output_path
        self.save_path = Path(repo_root / self.save_path).resolve()

    def connect_to_database(self):
        """
        Establish a connection to the WRDS database.

        Creates a connection using the provided username and password, and generates
        a pgpass file for authentication.
        """
        log_function_start("connect_to_database")
        self.logger.info("Connecting to WRDS database")
        self.db = wrds.Connection(
            wrds_username=self.username, wrds_password=self.password
        )
        self.db.create_pgpass_file()
        self.logger.info("Successfully connected to WRDS database")
        log_function_end("connect_to_database")

    def load_data(self):
        """
        Load data from the WRDS database using an SQL query.

        Reads the SQL query from a file and executes it against the database.

        Returns:
            pandas.DataFrame: The data retrieved from the SQL query.
        """
        log_function_start("load_data")
        self.logger.info("Loading data from OptionMetrics")
        self.logger.info("Executing SQL query from %s", self.sql_query_path)

        with open(self.sql_query_path, "r", encoding="utf-8") as file:
            sql_query = file.read()

        df = self.db.raw_sql(sql_query)

        if df.empty:
            self.logger.warning("No data returned from SQL query")

        log_function_end("load_data")
        return df

    def save_data(self, df):
        """
        Save the provided DataFrame to a CSV file.

        Args:
            df (pandas.DataFrame): The DataFrame to save.
        """
        log_function_start("save_data")
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Saving data to %s", self.save_path)
        df.to_csv(self.save_path, index=False)
        self.logger.info("Data saved successfully")
        log_function_end("save_data")


@hydra.main(version_base=None, config_path="../configs", config_name="data_loader")
def main(cfg: DictConfig):
    """
    Main function to orchestrate the data loading process.

    Initialises logging, creates a DataLoader instance, connects to the database,
    loads data, and saves it to a CSV file.

    Args:
        cfg (Optional[DictConfig]): Hydra configuration object, defaults to None.
    """
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")
    logger.info("Starting data loading process")
    data_loader = DataLoader(cfg)
    data_loader.connect_to_database()
    df = data_loader.load_data()
    data_loader.save_data(df)


if __name__ == "__main__":
    main()
