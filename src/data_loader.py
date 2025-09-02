import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig
import hydra
import wrds
from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end

load_dotenv()

class DataLoader:
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.db = None
        self.logger = get_logger("DataLoader")
        self.username = os.getenv("WRDS_USERNAME")
        self.password = os.getenv("WRDS_PASSWORD")
        self.sql_query = self.cfg.data_loader.sql_query
        src_dir = Path(__file__).parent
        self.sql_query_path = Path(src_dir / self.sql_query).resolve()
        repo_root = src_dir.parent
        self.save_path = self.cfg.data_loader.output_path
        self.save_path = Path(repo_root / self.save_path).resolve()

    def connect_to_database(self):
        log_function_start("connect_to_database")
        self.logger.info("Connecting to WRDS database")
        self.db = wrds.Connection(wrds_username=self.username, wrds_password=self.password)
        self.db.create_pgpass_file()
        self.logger.info("Successfully connected to WRDS database")
        log_function_end("connect_to_database")

    def load_data(self):
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
        log_function_start("save_data")
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Saving data to %s", self.save_path)
        df.to_csv(self.save_path, index=False)
        self.logger.info("Data saved successfully")
        log_function_end("save_data")


@hydra.main(version_base=None, config_path="../configs", config_name="data_loader")
def main(cfg: Optional[DictConfig] = None):
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")
    logger.info("Starting data loading process")
    data_loader = DataLoader(cfg)
    data_loader.connect_to_database()
    df = data_loader.load_data()
    data_loader.save_data(df)

if __name__ == "__main__":
    main()
