from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer

from ..core.tokenizer import inspect_tokenizer, train_or_import_tokenizer
from ..utils.mlflow_utils import (
    _parse_tags,
    log_artifact,
    log_metrics,
    log_params,
    start_tracking,
)


class ExitCode(int):
    OK = 0
    INVALID_INPUT = 2
    IO_ERROR = 3
    UNKNOWN = 4


tokenizer_app = typer.Typer(help="Tokenizer utilities")


@tokenizer_app.command("train")
def tokenizer_train(
    dataset_manifest: Path = typer.Option(
        ..., "--dataset-manifest", help="Path to dataset.manifest.json"
    ),
    algo: str = typer.Option(..., "--algo", help="Tokenization algorithm: char|bpe|unigram"),
    vocab_size: int = typer.Option(8000, "--vocab-size", help="Target vocabulary size"),
    norm: str = typer.Option("nfc", "--norm", help="Text normalization: none|nfc"),
    lower: bool = typer.Option(True, "--lower/--no-lower", help="Toggle lower casing"),
    punct_policy: str = typer.Option("keep", "--punct-policy", help="keep|strip|space"),
    seed: int = typer.Option(13, "--seed", help="Deterministic random seed"),
    out_dir: Path = typer.Option(Path("out/tokenizer"), "--out", help="Output directory"),
    use_external_tokenizer: Optional[Path] = typer.Option(
        None, "--use-external-tokenizer", help="Import an existing tokenizer.json"
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
        result = train_or_import_tokenizer(
            dataset_manifest=dataset_manifest,
            algo=algo,
            vocab_size=vocab_size,
            norm=norm,
            lower=lower,
            punct_policy=punct_policy,
            seed=seed,
            out_dir=out_dir,
            external_tokenizer=use_external_tokenizer,
        )
    except FileNotFoundError as exc:  # pragma: no cover - exercised via CLI integration tests
        typer.echo(f"[IO] {exc}")
        raise typer.Exit(code=ExitCode.IO_ERROR)
    except ValueError as exc:  # pragma: no cover
        typer.echo(f"[INVALID] {exc}")
        raise typer.Exit(code=ExitCode.INVALID_INPUT)
    except Exception as exc:  # pragma: no cover - ensures stable CLI exit codes
        typer.echo(f"[UNKNOWN] {exc}")
        raise typer.Exit(code=ExitCode.UNKNOWN)

    tracking_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI") or "file:./out/mlruns"
    derived_run_name = run_name or f"tokenizer.train[{algo}:{vocab_size}]"
    tags = {"ttlab.command": "tokenizer.train"}
    tags.update(_parse_tags(tag))

    manifest = result.get("manifest", {})
    report = result.get("report", {})
    summary = report.get("summary", {}) if isinstance(report, dict) else {}

    coverage = None
    coverage_per_task = summary.get("coverage_per_task") if isinstance(summary, dict) else None
    if isinstance(coverage_per_task, dict) and coverage_per_task:
        coverage = sum(float(value) for value in coverage_per_task.values()) / len(coverage_per_task)

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
            params_payload = {
                "dataset_manifest": str(dataset_manifest),
                "algo": algo,
                "vocab_size": vocab_size,
                "norm": norm,
                "lower": lower,
                "punct_policy": punct_policy,
                "seed": seed,
                "out": {"dir": str(out_dir)},
            }
            if use_external_tokenizer is not None:
                params_payload["use_external_tokenizer"] = str(use_external_tokenizer)
            log_params(params_payload)

            metrics_payload = {
                "vocab.size": manifest.get("vocab_size", vocab_size),
            }
            if coverage is not None:
                metrics_payload["coverage"] = coverage
            if "oov_rate" in summary:
                metrics_payload["unk.rate"] = summary["oov_rate"]
            log_metrics(metrics_payload)

            tokenizer_path = Path(result.get("tokenizer_path", out_dir / "tokenizer.json"))
            manifest_path = Path(result.get("manifest_path", out_dir / "tokenizer.manifest.json"))
            report_path = Path(result.get("report_path", out_dir / "tokenizer.report.json"))
            if tokenizer_path.exists():
                log_artifact(tokenizer_path, artifact_path="tokenizer")
            if manifest_path.exists():
                log_artifact(manifest_path, artifact_path="tokenizer")
            if report_path.exists():
                log_artifact(report_path, artifact_path="tokenizer")

            mlflow_run_id = getattr(getattr(active_run, "info", None), "run_id", None)

    if mlflow_run_id:
        result["mlflow_run_id"] = mlflow_run_id

    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    return ExitCode.OK


@tokenizer_app.command("inspect")
def tokenizer_inspect(
    tokenizer_path: Path = typer.Option(..., "--tokenizer", help="Path to tokenizer.json"),
    dataset_manifest: Path = typer.Option(..., "--dataset-manifest", help="Dataset manifest"),
    out_dir: Path = typer.Option(Path("out/tokenizer"), "--out", help="Output directory"),
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
) -> None:
    try:
        report = inspect_tokenizer(
            tokenizer_path=tokenizer_path, dataset_manifest=dataset_manifest, out_dir=out_dir
        )
    except FileNotFoundError as exc:  # pragma: no cover
        typer.echo(f"[IO] {exc}")
        raise typer.Exit(code=ExitCode.IO_ERROR)
    except ValueError as exc:  # pragma: no cover
        typer.echo(f"[INVALID] {exc}")
        raise typer.Exit(code=ExitCode.INVALID_INPUT)
    except Exception as exc:  # pragma: no cover
        typer.echo(f"[UNKNOWN] {exc}")
        raise typer.Exit(code=ExitCode.UNKNOWN)

    tracking_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI") or "file:./out/mlruns"
    derived_run_name = run_name or f"tokenizer.inspect[{tokenizer_path.stem}]"
    tags = {"ttlab.command": "tokenizer.inspect"}
    tags.update(_parse_tags(tag))

    summary = report.get("report", {}).get("summary", {}) if isinstance(report, dict) else {}
    sample_count = summary.get("num_samples") if isinstance(summary, dict) else None

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
            log_params(  # parameters follow the tracking contract
                {
                    "tokenizer": {"path": str(tokenizer_path)},
                    "dataset_manifest": str(dataset_manifest),
                    "out": {"dir": str(out_dir)},
                }
            )

            metrics_payload = {}
            if sample_count is not None:
                metrics_payload["tokinspect.sample_count"] = sample_count
            if metrics_payload:
                log_metrics(metrics_payload)

            report_path = Path(report.get("report_path", out_dir / "tokenizer.report.json"))
            if report_path.exists():
                log_artifact(report_path, artifact_path="tokenizer")

            mlflow_run_id = getattr(getattr(active_run, "info", None), "run_id", None)

    if mlflow_run_id:
        report["mlflow_run_id"] = mlflow_run_id

    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    tokenizer_app()
