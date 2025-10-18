"""Dataset validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from pandas.api import types as ptypes

from ttlab.config import DatasetField, DatasetFormat


class DatasetValidationError(RuntimeError):
    """Raised when dataset validation fails."""


@dataclass
class FieldReport:
    """Per-field validation result."""

    name: str
    exists: bool
    dtype: str | None
    null_fraction: float | None
    errors: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "exists": self.exists,
            "dtype": self.dtype,
            "null_fraction": self.null_fraction,
            "errors": self.errors,
        }


@dataclass
class ValidationReport:
    """Overall validation summary."""

    rows: int
    columns: int
    fields: List[FieldReport]

    def to_dict(self) -> Dict[str, object]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "fields": [field.to_dict() for field in self.fields],
        }


def _load_dataframe(path: Path, data_format: DatasetFormat) -> pd.DataFrame:
    if data_format == DatasetFormat.JSONL:
        return pd.read_json(path, orient="records", lines=True)
    if data_format == DatasetFormat.PARQUET:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {data_format}")


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    expected_lower = expected.lower()
    if expected_lower in {"int", "int64", "integer"}:
        return ptypes.is_integer_dtype(series)
    if expected_lower in {"float", "float64", "double"}:
        return ptypes.is_float_dtype(series)
    if expected_lower in {"str", "string", "text"}:
        return ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series)
    if expected_lower in {"bool", "boolean"}:
        return ptypes.is_bool_dtype(series)
    return True


def validate_dataset(path: Path | str, schema: Iterable[DatasetField], data_format: DatasetFormat) -> ValidationReport:
    """Validate a dataset and return a :class:`ValidationReport`."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise DatasetValidationError(f"Input dataset not found: {dataset_path}")

    df = _load_dataframe(dataset_path, data_format)

    field_reports: List[FieldReport] = []
    errors: List[str] = []

    for field in schema:
        if field.name not in df.columns:
            msg = f"Missing column '{field.name}'"
            if field.required:
                errors.append(msg)
            field_reports.append(
                FieldReport(
                    name=field.name,
                    exists=False,
                    dtype=None,
                    null_fraction=None,
                    errors=[msg] if field.required else [],
                )
            )
            continue

        series = df[field.name]
        null_fraction = float(series.isna().mean()) if len(df) else 0.0
        field_errors: List[str] = []
        if field.required and null_fraction > 0:
            field_errors.append(
                f"Field '{field.name}' contains nulls ({null_fraction:.2%}) but is required"
            )
        if not _dtype_matches(series, field.type):
            field_errors.append(
                f"Field '{field.name}' expected type {field.type}, got {series.dtype}"
            )
        if field_errors:
            errors.extend(field_errors)
        field_reports.append(
            FieldReport(
                name=field.name,
                exists=True,
                dtype=str(series.dtype),
                null_fraction=null_fraction,
                errors=field_errors,
            )
        )

    if errors:
        raise DatasetValidationError("; ".join(errors))

    return ValidationReport(rows=len(df), columns=len(df.columns), fields=field_reports)
