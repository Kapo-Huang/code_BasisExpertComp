import logging
import os
from pathlib import Path
from datetime import datetime

def setup_logging(
    level: str | None = None,
    log_dir: str | Path | None = None,
    log_file: str | None = None,
) -> None:
    level_name = (level or os.getenv("INR_LOG_LEVEL", "INFO")).strip().upper()
    level_value = getLevelName(level_name)

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level_value,
            format="%(asctime)s | %(message)s",
        )
    else:
        root.setLevel(level_value)

    if log_dir is None:
        return

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"run_{timestamp}.log"
    log_path = log_dir_path / log_file

    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            return

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level_value)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

def getLevelName(level_name):
    """
    Resolve a logging level name to its integer value in a safe, non-deprecated way.
    Falls back to logging.INFO for unknown values.
    """
    # Prefer public API if available (Python 3.11+)
    try:
        names_map = logging.getLevelNamesMapping()
    except AttributeError:
        # Fallback to internal mapping or a minimal known mapping
        names_map = getattr(logging, "_nameToLevel", {
            "CRITICAL": logging.CRITICAL,
            "FATAL": logging.FATAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "NOTSET": logging.NOTSET,
        })
    if isinstance(level_name, str):
        return names_map.get(level_name.strip().upper(), logging.INFO)
    try:
        return int(level_name)
    except Exception:
        return logging.INFO
