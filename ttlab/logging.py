"""Logging helpers for ttlab."""

from __future__ import annotations

from typing import Optional

from loguru import logger
from rich.console import Console


def configure_logging(level: str = "INFO", *, rich_tracebacks: bool = True) -> None:
    """Configure loguru with a Rich sink."""

    logger.remove()
    console = Console(stderr=True)
    logger.add(
        console.print,
        colorize=True,
        level=level,
        diagnose=rich_tracebacks,
        backtrace=rich_tracebacks,
        serialize=False,
    )


def get_logger(name: Optional[str] = None):
    """Return a child logger with an optional name."""

    return logger.bind(name=name) if name else logger


__all__ = ["configure_logging", "get_logger", "logger"]
