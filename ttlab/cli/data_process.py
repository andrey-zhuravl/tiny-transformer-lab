"""Dataset processing CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

from ttlab.core.process import DatasetProcessingError, process_dataset
from ttlab.core.validate import (
    DATA_FORMAT_JSONL,
    DATA_FORMAT_PARQUET,
    ExitCode,
    DatasetValidationError,
    _read_schema,
)
from ttlab.utils.console import print_json
from ttlab.utils.metrics_writer import MetricsWriter

VALID_FORMATS = {DATA_FORMAT_JSONL, DATA_FORMAT_PARQUET}


def _normalise_format(value: str) -> str:
    candidate = value.strip().upper()
    if candidate not in VALID_FORMATS:
        raise DatasetValidationError(
            f"Unsupported dataset format '{value}'. Expected one of: {sorted(VALID_FORMATS)}"
        )
    return candidate


def _parse_splits(values: Iterable[str]) -> Mapping[str, float]:
    if not values:
        raise DatasetProcessingError("At least one split ratio must be provided")

    splits: dict[str, float] = {}
    for raw in values:
        name, _, ratio = raw.partition("=")
        if not name or not ratio:
            raise DatasetProcessingError(
                "Split specification must follow the pattern '<name>=<ratio>'"
            )
        try:
            splits[name.strip()] = float(ratio)
        except ValueError as exc:
            raise DatasetProcessingError(f"Invalid ratio for split '{name}': {ratio}") from exc

    return splits


def run_data_process(
    *,
    in_path: Path,
    schema_path: Path,
    data_format: str = DATA_FORMAT_JSONL,
    splits: Iterable[str] = (),
    seed: int = 13,
    output_dir: Path = Path("out"),
    metrics_dir: Optional[Path] = Path("out"),
    log_to_mlflow: bool = False,
) -> ExitCode:
    try:
        schema = _read_schema(schema_path)
        normalised_format = _normalise_format(data_format)
        parsed_splits = _parse_splits(splits)
        result = process_dataset(
            dataset_path=in_path,
            schema_path=schema_path,
            schema=schema,
            data_format=normalised_format,
            split_ratios=parsed_splits,
            seed=seed,
            output_dir=output_dir,
            log_to_mlflow=log_to_mlflow,
        )
    except FileNotFoundError as exc:
        print(str(exc), flush=True)
        return ExitCode.IO_ERROR
    except (DatasetValidationError, DatasetProcessingError) as exc:
        print(str(exc), flush=True)
        return ExitCode.INVALID_INPUT
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Unexpected error: {exc}", flush=True)
        return ExitCode.UNKNOWN

    metrics_payload = {
        "rows_total": result.stats.rows_total,
        "valid_ratio": (
            result.validation_report.rows_valid / result.validation_report.rows_total
            if result.validation_report.rows_total
            else 0.0
        ),
        "task_distribution": dict(result.stats.task_distribution),
        "noise_rate": result.stats.noise_rate,
        "rows_per_split": {artifact.name: artifact.rows for artifact in result.split_artifacts},
        "duration_sec": result.duration_sec,
    }

    summary = dict(result.manifest)
    summary.update(
        {
            "manifest_path": str(result.manifest_path),
            "stats_path": str(result.stats_path),
            "splits": {artifact.name: str(artifact.path) for artifact in result.split_artifacts},
            "mlflow_run_id": result.mlflow_run_id,
        }
    )
    print_json(summary)

    if metrics_dir is not None:
        writer = MetricsWriter(metrics_dir)
        writer.write(metrics_payload)

    return ExitCode.OK


__all__ = ["run_data_process"]
