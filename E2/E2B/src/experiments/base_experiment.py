from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):

    def __init__(self, config):
        self.config = config
        self.results_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def prepare_models(self):
        pass

    @abstractmethod
    def run_experiment(self) -> pd.DataFrame:
        pass

    def save_results(self, results_df: pd.DataFrame, filename: str = "raw_results.csv"):
        save_path = self.results_dir / filename
        results_df.to_csv(save_path, index=False)
        logger.info(f"Results saved to {save_path}")

    def load_results(self, filename: str = "raw_results.csv") -> pd.DataFrame:
        load_path = self.results_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Results file not found: {load_path}")
        return pd.read_csv(load_path)
