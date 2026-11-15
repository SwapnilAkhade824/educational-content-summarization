"""Structured logging configuration for the application."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logging(level: str = "INFO", logger_name: Optional[str] = None) -> logging.Logger:
    """Configure and return a logger with a structured, timestamped format.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        logger_name: Optional logger name; defaults to the application logger.

    Returns:
        The configured logger.
    """
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(logger_name or "educational_summarizer")
    logger.setLevel(resolved_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(resolved_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Avoid duplicate propagation to root
    logger.propagate = False
    return logger
