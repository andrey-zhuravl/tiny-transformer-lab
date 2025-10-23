"""Utilities for persisting validation metrics."""

from __future__ import annotations

import json
from pathlib import Path

from ttlab.core.validate import ValidationReport


def write_metrics(report: ValidationReport, output_dir: Path) -> Path:
    """Write the validation report to ``output_dir`` as JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report_validation.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
    return output_path


__all__ = ["write_metrics"]
