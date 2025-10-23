"""Console helper utilities."""

from __future__ import annotations

import json
from typing import Any, Mapping


def print_json(payload: Mapping[str, Any]) -> None:
    """Pretty-print a mapping as JSON to stdout."""

    formatted = json.dumps(payload, indent=2, sort_keys=True)
    print(formatted)


__all__ = ["print_json"]
