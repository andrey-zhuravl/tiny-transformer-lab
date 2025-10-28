"""Validation CLI command for datasets produced by toy-lang-lab."""

from __future__ import annotations

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
    in_path: Path = typer.Option(..., "--in", help="Path to in"),
    schema_path: Path= typer.Option(..., "--schema", help="Path to in"),
    data_format: str = typer.Option(DATA_FORMAT_JSONL, "--format", help="schema format"),
    metrics_dir: Optional[Path] = typer.Option(None, "--metrics-dir", "-m", dir_okay=True)
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
    print_json(payload)

    if metrics_dir is not None:
        try:
            write_metrics(report, metrics_dir)
        except OSError as exc:  # pragma: no cover - filesystem errors are logged but non fatal
            print(f"Failed to write metrics: {exc}")

    return ExitCode.OK if report.rows_invalid == 0 else ExitCode.INVALID_INPUT

if __name__ == "__main__":  # pragma: no cover
    validate_app()