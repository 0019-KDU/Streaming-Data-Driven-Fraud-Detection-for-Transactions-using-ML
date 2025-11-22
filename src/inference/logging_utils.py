"""
Logging utilities for fraud detection inference service.

Provides standardized logging setup across all modules.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with standardized formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom format string (optional)
        date_format: Custom date format (optional)

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def setup_logger_from_config(name: str, config) -> logging.Logger:
    """
    Setup logger using Config object.

    Args:
        name: Logger name
        config: Config object with logging settings

    Returns:
        Configured logger instance
    """
    return setup_logger(
        name=name,
        level=config.logging.level,
        log_format=config.logging.format,
        date_format=config.logging.date_format
    )
