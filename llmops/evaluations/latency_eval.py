import time
from functools import wraps
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def track_latency(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        latency = round((time.perf_counter() - start) * 1000, 2)
        logger.info(f"{func.__name__} completed in {latency}ms")
        return result
    return wrapper
