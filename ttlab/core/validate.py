"""Dataset validation utilities."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback if PyYAML is unavailable
    yaml = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pandas as _pandas  # type: ignore
except Exception:  # pragma: no cover - fallback if pandas is unavailable
    _pandas = None  # type: ignore


DATA_FORMAT_JSONL = "JSONL"
DATA_FORMAT_PARQUET = "PARQUET"
SUPPORTED_FORMATS = {DATA_FORMAT_JSONL, DATA_FORMAT_PARQUET}


class ExitCode(Enum):
    """CLI exit codes for dataset validation."""

    OK = 0
    INVALID_INPUT = 2
    IO_ERROR = 3
    UNKNOWN = 4


class DatasetValidationError(RuntimeError):
    """Raised when the dataset or schema fails validation."""


@dataclass(slots=True)
class ValidationError:
    """Represents a validation error for a dataset record."""

    row: int
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"row": self.row, "message": self.message}


@dataclass(slots=True)
class ValidationReport:
    """Summary of dataset validation results."""

    rows_total: int
    rows_valid: int
    rows_invalid: int
    errors: List[ValidationError] = field(default_factory=list)
    duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows_total": self.rows_total,
            "rows_valid": self.rows_valid,
            "rows_invalid": self.rows_invalid,
            "errors": [error.to_dict() for error in self.errors],
            "duration_sec": round(self.duration_sec, 6),
        }


Schema = Mapping[str, Any]
Record = MutableMapping[str, Any]


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_yaml_lines(
    lines: List[tuple[int, str]], start: int, indent: int
) -> tuple[Dict[str, Any], int]:
    data: Dict[str, Any] = {}
    index = start
    length = len(lines)

    while index < length:
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if content.startswith("- "):
            raise DatasetValidationError("Unexpected list item at mapping level")

        key, _, remainder = content.partition(":")
        key = key.strip()
        remainder = remainder.strip()
        index += 1

        if remainder:
            data[key] = _parse_scalar(remainder)
            continue

        if index >= length or lines[index][0] <= current_indent:
            data[key] = {}
            continue

        next_indent, next_content = lines[index]
        if next_content.startswith("- "):
            values, index = _parse_yaml_list(lines, index, next_indent)
            data[key] = values
        else:
            nested, index = _parse_yaml_lines(lines, index, next_indent)
            data[key] = nested

    return data, index


def _parse_yaml_list(
    lines: List[tuple[int, str]], start: int, indent: int
) -> tuple[List[Any], int]:
    values: List[Any] = []
    index = start
    length = len(lines)

    while index < length:
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if not content.startswith("- "):
            break

        remainder = content[2:].strip()
        index += 1

        if remainder:
            values.append(_parse_scalar(remainder))
            continue

        if index >= length or lines[index][0] <= current_indent:
            values.append({})
            continue

        next_indent, next_content = lines[index]
        if next_content.startswith("- "):
            nested_list, index = _parse_yaml_list(lines, index, next_indent)
            values.append(nested_list)
        else:
            nested_map, index = _parse_yaml_lines(lines, index, next_indent)
            values.append(nested_map)

    return values, index


def _fallback_yaml_load(raw_text: str) -> Mapping[str, Any]:
    prepared: List[tuple[int, str]] = []
    for line in raw_text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        prepared.append((indent, line.strip()))

    if not prepared:
        return {}

    mapping, index = _parse_yaml_lines(prepared, 0, prepared[0][0])
    if index != len(prepared):
        raise DatasetValidationError("Failed to parse YAML schema")
    return mapping


def _read_schema(schema_path: Path) -> Schema:
    """Read a dataset schema definition from YAML."""

    with schema_path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read()

    if yaml is not None:  # pragma: no cover - exercised when PyYAML is available
        schema = yaml.safe_load(raw_text)
    else:
        schema = _fallback_yaml_load(raw_text)

    if not isinstance(schema, Mapping):
        raise DatasetValidationError("Schema root must be a mapping")

    if "dataset" not in schema or "task" not in schema:
        raise DatasetValidationError("Schema must define 'dataset' and 'task' sections")

    return schema


def _iter_jsonl_records(dataset_path: Path) -> Iterator[Record]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"Malformed JSON on line {line_number}: {exc}") from exc

            if not isinstance(data, MutableMapping):
                raise DatasetValidationError(
                    f"Expected JSON object on line {line_number}, received {type(data).__name__}"
                )

            yield data


def _iter_parquet_records(dataset_path: Path) -> Iterator[Record]:
    if _pandas is None:  # pragma: no cover - triggered in environments without pandas
        raise DatasetValidationError("Parquet support requires pandas to be installed")

    frame = _pandas.read_parquet(dataset_path)
    for row in frame.to_dict(orient="records"):
        if not isinstance(row, MutableMapping):
            raise DatasetValidationError("Parquet row must be a mapping")
        yield row


def _iter_records(dataset_path: Path, data_format: str) -> Iterator[Record]:
    if data_format == DATA_FORMAT_JSONL:
        yield from _iter_jsonl_records(dataset_path)
    elif data_format == DATA_FORMAT_PARQUET:
        yield from _iter_parquet_records(dataset_path)
    else:  # pragma: no cover - defensive guard
        raise DatasetValidationError(f"Unsupported data format: {data_format}")


def _validate_schema(schema: Schema) -> None:
    dataset_section = schema.get("dataset")
    if not isinstance(dataset_section, Mapping):
        raise DatasetValidationError("Schema 'dataset' section must be a mapping")

    task_section = schema.get("task")
    if not isinstance(task_section, Mapping):
        raise DatasetValidationError("Schema 'task' section must be a mapping")
    required_task = task_section.get("required", [])
    if not required_task:
        raise DatasetValidationError("Schema 'task.required' section must be an iterable of strings")

    required_fields = dataset_section.get("required_fields", [])
    if not isinstance(required_fields, Iterable):
        raise DatasetValidationError("'required_fields' must be an iterable of strings")

    meta_fields = dataset_section.get("meta_fields", {})
    if meta_fields and not isinstance(meta_fields, Mapping):
        raise DatasetValidationError("'meta_fields' must be a mapping")

    if meta_fields:
        required_meta = meta_fields.get("required", [])
        if not isinstance(required_meta, Iterable):
            raise DatasetValidationError("'meta_fields.required' must be an iterable of strings")


def _validate_record(record: Record, schema: Schema, row_number: int) -> List[ValidationError]:
    errors: List[ValidationError] = []
    dataset_section = schema["dataset"]
    task_section = schema["task"]

    required_fields = dataset_section.get("required_fields", [])
    for field in required_fields:
        if field not in record:
            errors.append(ValidationError(row=row_number, message=f"Missing required field '{field}'"))

    meta_fields = dataset_section.get("meta_fields", {})
    required_meta = meta_fields.get("required", []) if isinstance(meta_fields, Mapping) else []
    meta_value = record.get("meta")
    if required_meta:
        if not isinstance(meta_value, Mapping):
            errors.append(ValidationError(row=row_number, message="'meta' must be a mapping"))
        else:
            for meta_key in required_meta:
                if meta_key not in meta_value:
                    errors.append(
                        ValidationError(
                            row=row_number, message=f"Missing meta field '{meta_key}'"
                        )
                    )

    task_value = record.get("task")
    if task_value not in task_section.get("required", []):
        errors.append(ValidationError(row=row_number, message=f"Missing task field '{task_value}', it must be '{task_section.get('required')}'"))

    return errors


def validate_dataset(dataset_path: Path, schema: Schema, data_format: str) -> ValidationReport:
    """Validate a dataset against the provided schema."""

    if data_format not in SUPPORTED_FORMATS:
        raise DatasetValidationError(
            f"Unsupported dataset format '{data_format}'. Expected one of: {sorted(SUPPORTED_FORMATS)}"
        )

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    _validate_schema(schema)

    start_time = time.perf_counter()
    rows_total = 0
    rows_valid = 0
    errors: List[ValidationError] = []

    for row_number, record in enumerate(_iter_records(dataset_path, data_format), start=1):
        rows_total += 1
        row_errors = _validate_record(record, schema, row_number)
        if row_errors:
            errors.extend(row_errors)
        else:
            rows_valid += 1

    rows_invalid = rows_total - rows_valid
    duration = time.perf_counter() - start_time

    return ValidationReport(
        rows_total=rows_total,
        rows_valid=rows_valid,
        rows_invalid=rows_invalid,
        errors=errors,
        duration_sec=duration,
    )


__all__ = [
    "DATA_FORMAT_JSONL",
    "DATA_FORMAT_PARQUET",
    "DatasetValidationError",
    "ExitCode",
    "ValidationError",
    "ValidationReport",
    "_read_schema",
    "validate_dataset",
]
