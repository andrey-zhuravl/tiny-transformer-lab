"""Dataset statistics helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from ttlab.config import DatasetFormat


def _load_dataframe(path: Path, data_format: DatasetFormat) -> pd.DataFrame:
    if data_format == DatasetFormat.JSONL:
        return pd.read_json(path, orient="records", lines=True)
    if data_format == DatasetFormat.PARQUET:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {data_format}")


def compute_stats(
    path: Path | str,
    data_format: DatasetFormat,
    *,
    text_fields: Optional[Iterable[str]] = None,
    key_field: str | None = None,
) -> Dict[str, object]:
    """Compute basic statistics about a dataset."""

    dataset_path = Path(path)
    df = _load_dataframe(dataset_path, data_format)
    text_fields = set(text_fields or [])

    per_column: Dict[str, Dict[str, float]] = {}
    for column in df.columns:
        series = df[column]
        null_fraction = float(series.isna().mean()) if len(df) else 0.0
        column_stats: Dict[str, float] = {"null_fraction": null_fraction}
        if column in text_fields and not series.empty:
            lengths = series.dropna().astype(str).str.len()
            if not lengths.empty:
                column_stats.update(
                    {
                        "text_length_mean": float(lengths.mean()),
                        "text_length_std": float(lengths.std(ddof=0)) if len(lengths) > 1 else 0.0,
                        "text_length_p95": float(lengths.quantile(0.95)),
                    }
                )
        per_column[column] = column_stats

    stats: Dict[str, object] = {
        "path": str(dataset_path),
        "rows": len(df),
        "columns": len(df.columns),
        "columns_stats": per_column,
    }
    if key_field and key_field in df.columns:
        stats["unique_keys"] = int(df[key_field].nunique(dropna=True))

    return stats
