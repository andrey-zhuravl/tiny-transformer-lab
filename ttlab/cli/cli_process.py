from __future__ import annotations

import enum
import functools
import os
from pathlib import Path
from typing import Mapping, Optional

import typer

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
from ttlab.utils.mlflow_utils import (
    _parse_tags,
    log_artifact,
    log_metrics,
    log_params,
    start_tracking,
)

process_app = typer.Typer(help="Data processing utilities")

VALID_FORMATS = {DATA_FORMAT_JSONL, DATA_FORMAT_PARQUET}


def _normalise_format(value: str) -> str:
    candidate = value.strip().upper()
    if candidate not in VALID_FORMATS:
        raise DatasetValidationError(
            f"Unsupported dataset format '{value}'. Expected one of: {sorted(VALID_FORMATS)}"
        )
    return candidate


def _parse_splits(values: str) -> Mapping[str, float]:
    if not values:
        raise DatasetProcessingError("At least one split ratio must be provided")

    splits: dict[str, float] = {}
    for raw in values.split(sep=","):
        name, _, ratio = raw.partition("=")
        if not name or not ratio:
            raise DatasetProcessingError(
                "Split specification must follow the pattern '<name>=<ratio>'"
            )
        try:
            splits[name.strip()] = float(ratio)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise DatasetProcessingError(f"Invalid ratio for split '{name}': {ratio}") from exc

    return splits


@process_app.command("run")
def run_data_process(
    in_path: Path = typer.Option(..., "--in", help="Path to input dataset"),
    schema_path: Path = typer.Option(..., "--schema", help="Path to schema"),
    data_format: str = typer.Option(DATA_FORMAT_JSONL, "--format", help="Dataset format"),
    split: str = typer.Option(..., "--split", help="Comma-separated split ratios"),
    seed: int = typer.Option(13, "--seed", help="Deterministic random seed"),
    output_dir: Path = typer.Option(..., "--out", help="Path to output directory"),
    metrics_dir: Optional[Path] = typer.Option(
        None, "--metrics", help="Optional directory for metrics output"
    ),
    mlflow: bool = typer.Option(False, "--mlflow", help="Enable MLflow tracking"),
    mlflow_uri: Optional[str] = typer.Option(
        None,
        "--mlflow-uri",
        help="MLflow tracking URI (default: env MLFLOW_TRACKING_URI or file:./out/mlruns)",
    ),
    experiment: str = typer.Option("ttlab.dev", "--experiment", help="MLflow experiment name"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Override MLflow run name"),
    parent_run_id: Optional[str] = typer.Option(
        None, "--parent-run-id", help="Parent MLflow run id for nested runs"
    ),
    tag: list[str] = typer.Option([], "--tag", help="Additional MLflow tag as key=value"),
) -> ExitCode:
    try:
        schema = _read_schema(schema_path)
        normalised_format = _normalise_format(data_format)
        parsed_splits = _parse_splits(split)
        result = process_dataset(
            dataset_path=in_path,
            schema_path=schema_path,
            schema=schema,
            data_format=normalised_format,
            split_ratios=parsed_splits,
            seed=seed,
            output_dir=output_dir,
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

    if metrics_dir is not None:
        writer = MetricsWriter(metrics_dir)
        writer.write(metrics_payload)

    tracking_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI") or "file:./out/mlruns"
    derived_run_name = run_name or f"process[{in_path.stem}]"
    tags = {"ttlab.command": "process.run"}
    tags.update(_parse_tags(tag))

    split_rows = {artifact.name: artifact.rows for artifact in result.split_artifacts}
    mlflow_run_id: Optional[str] = None
    with start_tracking(
        mlflow,
        experiment,
        derived_run_name,
        uri=tracking_uri,
        parent_run_id=parent_run_id,
        tags=tags,
    ) as active_run:
        if active_run is not None:
            log_params(
                {
                    "schema": {"path": str(schema_path)},
                    "format": normalised_format,
                    "split": split,
                    "seed": seed,
                    "out": {"dir": str(output_dir)},
                }
            )
            metric_payload = {"rows.total": result.stats.rows_total}
            metric_payload.update({f"rows.{name}": count for name, count in split_rows.items()})
            metric_payload.update(
                {
                    "rows.valid": result.validation_report.rows_valid,
                    "rows.invalid": result.validation_report.rows_invalid,
                }
            )
            log_metrics(metric_payload)
            log_artifact(result.manifest_path, artifact_path="process")
            log_artifact(result.stats_path, artifact_path="process")
            if metrics_dir is not None and metrics_dir.exists():
                log_artifact(metrics_dir, artifact_path="process/metrics_dir")
            mlflow_run_id = getattr(getattr(active_run, "info", None), "run_id", None)

    if mlflow_run_id:
        result.mlflow_run_id = mlflow_run_id

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

    return ExitCode.OK


def returns_exit_code(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        code = fn(*args, **kwargs)
        if code is None:
            return
        if isinstance(code, enum.Enum):
            code = code.value
        try:
            code = int(code)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Unsupported exit code returned: {code!r}") from exc
        raise typer.Exit(code=code)

    return wrapper


@process_app.command("run_test")
@returns_exit_code
def run_test(
    in_path: Path = typer.Option(..., "--in", help="Path to in"),
) -> ExitCode:
    return ExitCode.INVALID_INPUT


if __name__ == "__main__":  # pragma: no cover
    process_app()
