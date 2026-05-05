"""Tiny logger wrapper — one function so we don't reconfigure across modules."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_LOGGERS = {}


def get_logger(name: str = "memedit", level: Optional[int] = None) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level or logging.INFO)
    _LOGGERS[name] = logger
    return logger
