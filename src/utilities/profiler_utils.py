import time
import functools
import logging

logger = logging.getLogger(__name__)


def measure_time_taken(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Executing {func.__name__}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{func.__name__} executed in {duration:.4f} seconds")
        return result

    return wrapper
