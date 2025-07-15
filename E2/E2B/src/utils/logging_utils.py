import logging
import os
from datetime import datetime
from typing import Union

def _make_log_dir(path: Union[str, os.PathLike] = "logs") -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)

def setup_logging(name: str = __name__, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    _make_log_dir(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"run_{timestamp}.log")
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
        )
    return logging.getLogger(name)

