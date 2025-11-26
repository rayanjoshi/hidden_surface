import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
import hydra

from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end


class DateSelector:
    def __init__(self, cfg: DictConfig, data_path: Path, output_path: Path) -> None:
        self.cfg = cfg
        self.data_path = data_path
        self.output_path = output_path
        self.df = pd.read_csv(self.data_path, parse_dates=['date'])

    def select_data(self, date):
        """
        Selects rows from the dataframe of the specified date.

        Parameters:
        date (str): The date in 'YYYY-MM-DD' format.

        """
        log_function_start("select_data")
        if 'date' not in self.df.columns:
            raise ValueError("The dataframe does not contain a 'date' column.")
        
        selected_df = self.df.copy()
        selected_df.sort_index(inplace=True)
        selected_df = selected_df[selected_df['date'] == pd.to_datetime(date)]
        
        self._save_data(self.output_path, selected_df)
        
        
        log_function_end("select_data")
        
    def _save_data(self, output_path: Path, selected_df: pd.DataFrame) -> None:
        """
        Saves the selected dataframe to a CSV file.

        Parameters:
        output_path (Path): The path where the CSV file will be saved.
        selected_df (pd.DataFrame): The dataframe to be saved.
        """
        log_function_start("save_data")
        selected_df.to_csv(output_path, index=False)
        log_function_end("save_data")
        

@hydra.main(version_base=None, config_path="../configs", config_name="select_date")
def main(cfg: DictConfig) -> None:
    """
    Main function to execute the date selection process.

    Parameters:
    cfg (DictConfig): The configuration object containing parameters.
    """
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("DateSelector")
    logger.info("Starting date selection process")
    
    repo_root = Path(__file__).parent.parent

    data_path = (repo_root / cfg.data_processor.output_path).resolve()
    output_path = (repo_root / cfg.select_date.output_path).resolve()
    date_selector = DateSelector(cfg, data_path, output_path)
    date_selector.select_data(cfg.select_date.date)

    logger.info("Date selection process completed")

if __name__ == "__main__":
    main()
