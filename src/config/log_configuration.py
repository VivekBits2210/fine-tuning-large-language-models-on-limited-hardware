from typing import Optional
import logging

FORMAT = '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
logger = logging.getLogger()
formatter = logging.Formatter(FORMAT)


class LogConfig:
    @staticmethod
    def setup_logging():
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    @staticmethod
    def set_logging_level(level: str):
        level: Optional[int] = getattr(logging, level.upper(), None)
        if isinstance(level, int):
            logger.setLevel(level=level)
        else:
            raise ValueError(f"Invalid logging level: {level}!")
