"""Validation CLI command for datasets produced by toy-lang-lab."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ttlab.core.validate import (
    DATA_FORMAT_JSONL,
    DATA_FORMAT_PARQUET,
    ExitCode,
    DatasetValidationError,
    _read_schema,
    validate_dataset,
)
from ttlab.utils.console import print_json
from ttlab.utils.metrics_writer import write_metrics
from ttlab.utils.mlflow_utils import (
    _parse_tags,
    log_json,
    log_metrics,
    log_params,
    start_tracking,
)

import typer

validate_app = typer.Typer(help="Data validation utilities")

VALID_FORMATS = {DATA_FORMAT_JSONL, DATA_FORMAT_PARQUET}


def _normalise_format(value: str) -> str:
    candidate = value.strip().upper()
    if candidate not in VALID_FORMATS:
        raise DatasetValidationError(
            f"Unsupported dataset format '{value}'. Expected one of: {sorted(VALID_FORMATS)}"
        )
    return candidate

@validate_app.command("run2")
def run2_data_validate(
    in_path: Path = typer.Option(..., "--in", help="Path to in"),
) -> ExitCode:
    typer.echo("run2_data_validate")
    return ExitCode.OK

@validate_app.command("run")
def run_data_validate(
    in_path: Path = typer.Option(..., "--in", help="Path to dataset"),
    schema_path: Path = typer.Option(..., "--schema", help="Path to schema"),
    data_format: str = typer.Option(DATA_FORMAT_JSONL, "--format", help="Dataset format"),
    metrics_dir: Optional[Path] = typer.Option(
        None, "--metrics-dir", "-m", dir_okay=True, help="Directory to write metrics"
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
    """Execute the dataset validation routine and return an exit code."""
    try:
        schema = _read_schema(schema_path)
        normalised_format = _normalise_format(data_format)
        report = validate_dataset(in_path, schema, normalised_format)
    except FileNotFoundError as exc:
        print(str(exc), flush=True)
        return ExitCode.IO_ERROR
    except DatasetValidationError as exc:
        print(str(exc), flush=True)
        typer.echo(str(exc))
        return ExitCode.INVALID_INPUT
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Unexpected error: {exc}", flush=True)
        return ExitCode.UNKNOWN

    payload = report.to_dict()

    tracking_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI") or "file:./out/mlruns"
    derived_run_name = run_name or f"validate[{in_path.stem}]"
    tags = {"ttlab.command": "validate.run"}
    tags.update(_parse_tags(tag))

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
                    "in": {"path": str(in_path)},
                }
            )
            log_metrics(
                {
                    "records.total": payload["rows_total"],
                    "records.valid": payload["rows_valid"],
                    "records.invalid": payload["rows_invalid"],
                }
            )
            log_json(payload, "validate/report.json")
            mlflow_run_id = getattr(getattr(active_run, "info", None), "run_id", None)

    if mlflow_run_id:
        payload["mlflow_run_id"] = mlflow_run_id

    print_json(payload)

    if metrics_dir is not None:
        try:
            write_metrics(report, metrics_dir)
        except OSError as exc:  # pragma: no cover - filesystem errors are logged but non fatal
            print(f"Failed to write metrics: {exc}")

    return ExitCode.OK if report.rows_invalid == 0 else ExitCode.INVALID_INPUT

if __name__ == "__main__":  # pragma: no cover
    validate_app()