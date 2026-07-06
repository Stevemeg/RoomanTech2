"""Loguru-based logger setup."""

import os
import sys

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)


def get_logger(name: str | None = None):
    return logger.bind(module=name) if name else logger
