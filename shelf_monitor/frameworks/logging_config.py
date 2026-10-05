"""Structured logging setup."""
from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", fmt: str = "text", file_path: Optional[str] = None) -> None:
    """Configure root logger.  Call once at startup."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if file_path:
        handlers.append(logging.FileHandler(file_path, encoding="utf-8"))

    if fmt == "json":
        log_format = '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
    else:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
