"""Typer-based CLI entrypoint for ttlab."""

from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from ttlab.config import DatasetField, DatasetFormat, load_config
from ttlab.data import (
    DatasetValidationError,
    compute_stats,
    convert_dataset,
    split_dataset,
    validate_dataset,
)
from ttlab.logging import configure_logging, get_logger
from ttlab.mlflow_utils import log_dataset_run, ping_tracking_server
from ttlab.observability.metrics_writer import MetricsWriter

app = typer.Typer(help="Tiny Transformer Lab CLI")
console = Console()


class ExitCode(IntEnum):
    OK = 0
    INVALID_INPUT = 2
    IO_ERROR = 3
    MLFLOW_ERROR = 4
    UNKNOWN = 5


def _read_schema(path: Path) -> List[DatasetField]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    fields = data.get("schema_fields", data.get("fields", []))
    return [DatasetField.model_validate(field) for field in fields]


@app.callback()
def main(verbosity: int = typer.Option(0, "-v", "--verbose", count=True)) -> None:
    level = "INFO"
    if verbosity >= 2:
        level = "DEBUG"
    elif verbosity == 1:
        level = "INFO"
    configure_logging(level)


@app.command()
def init(
    config: Path = typer.Option(Path("conf/base.yaml"), help="Path to the base configuration"),
) -> None:
    """Create project directories defined in the configuration."""

    try:
        project_config = load_config(config)
    except Exception as exc:  # pragma: no cover - Typer prints message
        raise typer.Exit(code=ExitCode.INVALID_INPUT) from exc

    logger = get_logger("init")
    created = []
    for path in project_config.missing_paths():
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))
    if created:
        logger.info("Created directories: {}", ", ".join(created))
    else:
        logger.info("All configured directories already exist")


@app.command()
def check(
    config: Path = typer.Option(Path("conf/base.yaml"), help="Path to the base configuration"),
) -> None:
    """Run quick environment checks."""

    try:
        project_config = load_config(config)
    except Exception as exc:
        console.print(f"[red]Failed to load config: {exc}")
        raise typer.Exit(code=ExitCode.INVALID_INPUT)

    table = Table(title="Environment Check")
    table.add_column("Check")
    table.add_column("Status")

    missing = project_config.missing_paths()
    table.add_row("Configured directories", "OK" if not missing else f"Missing: {missing}")

    try:
        mlflow_info = ping_tracking_server(project_config.mlflow)
        table.add_row("MLflow", f"OK ({mlflow_info['experiment_id']})")
    except Exception as exc:  # pragma: no cover - depends on mlflow backend
        table.add_row("MLflow", f"Error: {exc}")
        console.print(table)
        raise typer.Exit(code=ExitCode.MLFLOW_ERROR)

    console.print(table)


@app.command(name="data:validate")
def data_validate(
    input_path: Path = typer.Option(..., "--in", help="Input dataset"),
    schema_path: Path = typer.Option(..., "--schema", help="Schema YAML"),
    data_format: DatasetFormat = typer.Option(DatasetFormat.JSONL, "--format"),
) -> None:
    """Validate a dataset using a schema."""

    try:
        schema = _read_schema(schema_path)
        report = validate_dataset(input_path, schema, data_format)
    except DatasetValidationError as exc:
        console.print(f"[red]Validation failed: {exc}")
        raise typer.Exit(code=ExitCode.INVALID_INPUT)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(code=ExitCode.IO_ERROR)
    except Exception as exc:  # pragma: no cover
        console.print(f"[red]Unexpected error: {exc}")
        raise typer.Exit(code=ExitCode.UNKNOWN)

    console.print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


@app.command(name="data:convert")
def data_convert(
    input_path: Path = typer.Option(..., "--in", help="Input dataset"),
    output_path: Path = typer.Option(..., "--out", help="Output dataset"),
    to: DatasetFormat = typer.Option(..., "--to", help="Target format"),
) -> None:
    """Convert datasets between JSONL and Parquet."""

    try:
        result = convert_dataset(input_path, output_path, to)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(code=ExitCode.IO_ERROR)
    except ValueError as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(code=ExitCode.INVALID_INPUT)

    console.print(json.dumps(result, indent=2))


@app.command(name="data:split")
def data_split(
    input_path: Path = typer.Option(..., "--in", help="Input dataset"),
    output_dir: Path = typer.Option(..., "--out", help="Output directory"),
    ratios: str = typer.Option("0.8,0.1,0.1", help="train,val,test ratios"),
    seed: int = typer.Option(42, help="Random seed"),
    strategy: str = typer.Option("by_row", help="Split strategy"),
    key_column: Optional[str] = typer.Option(None, help="Key column for by_key"),
) -> None:
    """Deterministically split a dataset."""

    try:
        ratios_tuple = tuple(float(val.strip()) for val in ratios.split(","))
        result = split_dataset(
            input_path,
            output_dir,
            ratios_tuple,
            seed=seed,
            strategy=strategy,
            key_column=key_column,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(code=ExitCode.IO_ERROR)
    except ValueError as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(code=ExitCode.INVALID_INPUT)

    console.print(json.dumps(result, indent=2))


@app.command(name="data:stats")
def data_stats(
    input_path: Path = typer.Option(..., "--in", help="Input dataset"),
    data_format: DatasetFormat = typer.Option(DatasetFormat.PARQUET, "--format"),
    key_field: Optional[str] = typer.Option(None, help="Key field"),
    text_fields: List[str] = typer.Option([], "--text-field", help="Text fields"),
    log_mlflow: bool = typer.Option(False, "--log-mlflow", help="Log to MLflow"),
    config: Path = typer.Option(Path("conf/base.yaml"), help="Base configuration"),
) -> None:
    """Compute dataset statistics and optionally log to MLflow."""

    stats = compute_stats(
        input_path,
        data_format,
        key_field=key_field,
        text_fields=text_fields,
    )
    console.print(json.dumps(stats, indent=2, ensure_ascii=False))

    if log_mlflow:
        try:
            project_config = load_config(config)
            run_id = log_dataset_run(
                project_config.mlflow,
                run_name=project_config.general.run_name,
                stats=stats,
                params={
                    "format": data_format.value,
                    "seed": project_config.general.seed,
                },
            )
        except Exception as exc:  # pragma: no cover
            console.print(f"[red]Failed to log to MLflow: {exc}")
            raise typer.Exit(code=ExitCode.MLFLOW_ERROR)
        console.print(f"Logged to MLflow run {run_id}")


@app.command(name="mlflow:ping")
def mlflow_ping(
    config: Path = typer.Option(Path("conf/base.yaml"), help="Base configuration"),
) -> None:
    """Check MLflow connectivity."""

    try:
        project_config = load_config(config)
        info = ping_tracking_server(project_config.mlflow)
    except Exception as exc:  # pragma: no cover
        console.print(f"[red]MLflow ping failed: {exc}")
        raise typer.Exit(code=ExitCode.MLFLOW_ERROR)

    console.print(json.dumps(info, indent=2))


@app.command()
def smoke(
    config: Path = typer.Option(Path("conf/base.yaml"), help="Base configuration"),
) -> None:
    """Run an end-to-end smoke test."""

    project_config = load_config(config)
    logger = get_logger("smoke")

    for path in project_config.missing_paths():
        path.mkdir(parents=True, exist_ok=True)

    metrics_path = project_config.paths.out_metrics / "metrics.jsonl"
    with MetricsWriter(metrics_path, default_run_id=project_config.general.run_name) as writer:
        writer.log(event="smoke", metric="smoke_start", value=1.0)

        schema = project_config.data.schema_fields
        try:
            validate_dataset(
                project_config.data.input_path,
                schema,
                project_config.data.format,
            )
        except DatasetValidationError as exc:
            writer.log(event="smoke", metric="validation_failed", value=1.0, labels={"error": str(exc)})
            writer.flush()
            raise typer.Exit(code=ExitCode.INVALID_INPUT)

        converted_path = project_config.paths.data_processed / "smoke.parquet"
        convert_dataset(project_config.data.input_path, converted_path, DatasetFormat.PARQUET)
        stats = compute_stats(converted_path, DatasetFormat.PARQUET, key_field="id", text_fields=["text"])
        writer.log(event="data_stats", metric="rows_total", value=float(stats["rows"]))

        try:
            run_id = log_dataset_run(
                project_config.mlflow,
                run_name=project_config.general.run_name,
                stats=stats,
                params={"format": DatasetFormat.PARQUET.value},
                artifacts={"dataset_schema": project_config.data.schema_dict()},
            )
            writer.log(event="smoke", metric="mlflow_run", value=1.0, labels={"run_id": run_id})
        except Exception as exc:  # pragma: no cover
            writer.log(event="smoke", metric="mlflow_failed", value=1.0, labels={"error": str(exc)})
            writer.flush()
            raise typer.Exit(code=ExitCode.MLFLOW_ERROR)

        writer.log(event="smoke", metric="smoke_done", value=1.0)

    logger.info("Smoke test completed")


if __name__ == "__main__":  # pragma: no cover
    app()
