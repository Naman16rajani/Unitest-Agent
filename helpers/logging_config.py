import logging
from pathlib import Path
from datetime import datetime

LOG_FILE = Path("debug.log")


def create_logger(name="UnitestAgent"):
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid adding handlers multiple times
        return logger

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(f"./logs/{name}_{timestamp}.log")
    filename.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(filename=filename)
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


logger = create_logger()
