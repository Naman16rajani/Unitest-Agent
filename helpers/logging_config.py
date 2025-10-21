import logging
from pathlib import Path

LOG_FILE = Path("debug.log")


def _create_logger():
    logger = logging.getLogger("UnitestAgent")
    if logger.handlers:  # avoid adding handlers multiple times
        return logger

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(str(LOG_FILE))
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


logger = _create_logger()
