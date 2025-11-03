"""Model package exposing registration utilities and built-in models."""

from __future__ import annotations

from .base import BaseTTLabModel
from .registry import create_model, list_models, register_model

# Import modules that register models on import.
from . import vanilla  # noqa: F401  # pylint: disable=unused-import

__all__ = [
    "BaseTTLabModel",
    "create_model",
    "list_models",
    "register_model",
]
