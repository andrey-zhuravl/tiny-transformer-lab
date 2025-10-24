"""Utilities for persisting validation and processing metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ttlab.core.validate import ValidationReport


def write_metrics(report: ValidationReport, output_dir: Path) -> Path:
    """Write the validation report to ``output_dir`` as JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report_validation.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
    return output_path


class MetricsWriter:
    """Append-only JSONL metrics writer.

    The writer stores each metrics payload on a separate line, allowing
    downstream tooling to stream and aggregate results efficiently.
    """

    def __init__(self, output_dir: Path, filename: str = "metrics.jsonl") -> None:
        self.output_dir = output_dir
        self.filename = filename
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / self.filename

    def write(self, payload: Mapping[str, Any]) -> Path:
        """Append ``payload`` to the metrics file and return the path."""

        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, sort_keys=True)
            handle.write("\n")
        return self.path


__all__ = ["MetricsWriter", "write_metrics"]
