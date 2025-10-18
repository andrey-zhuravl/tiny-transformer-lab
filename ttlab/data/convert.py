"""Dataset conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from ttlab.config import DatasetFormat


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(path, orient="records", lines=True)
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format for {path}")


def convert_dataset(
    src: Path | str,
    dst: Path | str,
    to_format: DatasetFormat | str,
    *,
    compression: str = "snappy",
) -> Dict[str, object]:
    """Convert a dataset between JSONL and Parquet."""

    src_path = Path(src)
    dst_path = Path(dst)
    target_format = (
        to_format if isinstance(to_format, DatasetFormat) else DatasetFormat(to_format.upper())
    )

    if not src_path.exists():
        raise FileNotFoundError(src_path)
    if src_path.resolve() == dst_path.resolve():
        raise ValueError("Source and destination paths must differ")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    df = _read_any(src_path)
    if target_format == DatasetFormat.PARQUET:
        df.to_parquet(dst_path, index=False, compression=compression)
    elif target_format == DatasetFormat.JSONL:
        df.to_json(dst_path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported target format {target_format}")

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "destination": str(dst_path),
        "format": target_format.value,
    }
