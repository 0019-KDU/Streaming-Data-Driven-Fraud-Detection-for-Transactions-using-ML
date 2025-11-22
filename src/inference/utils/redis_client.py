"""
Shared Redis connection pool for velocity and ATO services.

Provides a singleton Redis client with connection pooling.
"""

import redis
from typing import Optional
from ..logging_utils import setup_logger

logger = setup_logger(__name__)

_redis_client: Optional[redis.Redis] = None


def get_redis_client(config) -> redis.Redis:
    """
    Get or create a Redis client with connection pooling.

    Args:
        config: Config object with Redis settings

    Returns:
        Redis client instance
    """
    global _redis_client

    if _redis_client is None:
        logger.info(
            f"Initializing Redis client: {config.redis.host}:{config.redis.port}, "
            f"db={config.redis.db}, max_connections={config.redis.max_connections}"
        )

        pool = redis.ConnectionPool(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            password=config.redis.password,
            max_connections=config.redis.max_connections,
            socket_timeout=config.redis.socket_timeout,
            decode_responses=True  # Automatically decode bytes to strings
        )

        _redis_client = redis.Redis(connection_pool=pool)

        # Test connection
        try:
            _redis_client.ping()
            logger.info("Redis connection established successfully")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    return _redis_client


def close_redis_client() -> None:
    """Close the Redis client connection pool."""
    global _redis_client
    if _redis_client:
        _redis_client.close()
        _redis_client = None
        logger.info("Redis client closed")
