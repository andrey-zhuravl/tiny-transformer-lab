"""Utility for writing metrics to JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


@dataclass
class MetricsWriter:
    """Append-only metrics writer that stores events in JSONL format."""

    path: Path
    default_run_id: Optional[str] = None
    _buffer: list[Dict[str, object]] = field(default_factory=list)

    def log(
        self,
        *,
        event: str,
        metric: str,
        value: float,
        labels: Optional[Dict[str, object]] = None,
        run_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        payload = {
            "ts": (timestamp or datetime.now(timezone.utc)).isoformat(),
            "event": event,
            "metric": metric,
            "value": value,
            "labels": labels or {},
            "run_id": run_id or self.default_run_id or "",
        }
        self._buffer.append(payload)

    def flush(self) -> None:
        if not self._buffer:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for record in self._buffer:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._buffer.clear()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.flush()


__all__ = ["MetricsWriter"]
