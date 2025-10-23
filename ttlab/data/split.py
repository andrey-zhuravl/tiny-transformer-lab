"""Deterministic dataset splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple, cast

import numpy as np
import pandas as pd

from ttlab.config.models import DatasetFormat


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(path, orient="records", lines=True)
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format for {path}")


def _write_parquet(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)


def _compute_boundaries(length: int, ratios: Tuple[float, float, float]) -> Tuple[int, int]:
    train_end = int(ratios[0] * length)
    val_end = train_end + int(ratios[1] * length)
    return train_end, val_end


def split_dataset(
    src: Path | str,
    output_dir: Path | str,
    ratios: Iterable[float],
    *,
    seed: int = 42,
    strategy: str = "by_row",
    key_column: str | None = None,
) -> Dict[str, object]:
    """Split a dataset into train/validation/test partitions."""

    src_path = Path(src)
    df = _read_any(src_path)
    ratios_tuple = tuple(ratios)
    if not np.isclose(sum(ratios_tuple), 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    if len(ratios_tuple) != 3:
        raise ValueError("Expected three ratios for train/val/test")

    ratios_cast = cast(Tuple[float, float, float], ratios_tuple)
    rng = np.random.default_rng(seed)

    if strategy == "by_key":
        if not key_column:
            raise ValueError("key_column is required for by_key strategy")
        if key_column not in df.columns:
            raise ValueError(f"Key column '{key_column}' not found")
        unique_keys = df[key_column].dropna().unique()
        rng.shuffle(unique_keys)
        train_end, val_end = _compute_boundaries(len(unique_keys), ratios_cast)
        train_keys = set(unique_keys[:train_end])
        val_keys = set(unique_keys[train_end:val_end])
        splits = {
            "train": df[df[key_column].isin(train_keys)],
            "val": df[df[key_column].isin(val_keys)],
            "test": df[~df[key_column].isin(train_keys | val_keys)],
        }
    elif strategy == "by_row":
        indices = np.arange(len(df))
        rng.shuffle(indices)
        train_end, val_end = _compute_boundaries(len(df), ratios_cast)
        splits = {
            "train": df.iloc[indices[:train_end]],
            "val": df.iloc[indices[train_end:val_end]],
            "test": df.iloc[indices[val_end:]],
        }
    else:
        raise ValueError(f"Unsupported strategy {strategy}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {}
    for split_name, split_df in splits.items():
        destination = out_dir / f"{split_name}.parquet"
        _write_parquet(split_df, destination)
        stats[f"{split_name}_rows"] = len(split_df)

    stats.update({
        "source": str(src_path),
        "output_dir": str(out_dir),
        "format": DatasetFormat.PARQUET.value,
    })
    return stats
