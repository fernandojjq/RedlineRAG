"""Lightweight logging helper.

We don't pull in loguru or structlog for this - Python's stdlib logging with
a clean format is more than enough for a CLI tool.
"""

from __future__ import annotations

import logging
import sys
from typing import Final


_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger exactly once.

    Calling this multiple times in a session is safe: handlers are reset so
    we never end up with duplicate log lines.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Cheap; safe to call from anywhere."""
    return logging.getLogger(name)
