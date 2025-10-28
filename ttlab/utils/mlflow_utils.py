from __future__ import annotations

import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency
    import mlflow
except Exception:  # pragma: no cover - mlflow remains optional
    mlflow = None  # type: ignore[assignment]


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested mapping using dot-separated keys."""

    items: Dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items |= _flatten(value, name)
        else:
            items[name] = value
    return items


def _parse_tags(pairs: Iterable[str]) -> Dict[str, str]:
    """Parse ``key=value`` pairs provided on the command line into a mapping."""

    tags: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        tags[key.strip()] = value.strip()
    return tags


def _git_commit() -> Optional[str]:
    """Return the current git commit if available."""

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
    except Exception:  # pragma: no cover - git optional at runtime
        return None


@contextmanager
def start_tracking(
    enabled: bool,
    experiment: str,
    run_name: str,
    *,
    uri: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
):
    """Context manager that yields an active MLflow run when tracking is enabled."""

    if not enabled or mlflow is None:
        yield None
        return

    if uri:
        mlflow.set_tracking_uri(uri)

    mlflow.set_experiment(experiment)

    merged_tags = {
        "host": socket.gethostname(),
        "git.commit": _git_commit() or "",
    }
    if tags:
        merged_tags |= tags

    if parent_run_id:
        with mlflow.start_run(run_id=parent_run_id):
            with mlflow.start_run(run_name=run_name, nested=True, tags=merged_tags) as run:
                yield run
    else:
        with mlflow.start_run(run_name=run_name, tags=merged_tags) as run:
            yield run


def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to the active MLflow run."""

    if mlflow is None:
        return

    flattened = {
        key: (value if isinstance(value, (int, float, bool)) else str(value))
        for key, value in _flatten(params).items()
    }
    mlflow.log_params(flattened)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to the active MLflow run."""

    if mlflow is None:
        return

    numeric_metrics = {key: float(value) for key, value in metrics.items()}
    mlflow.log_metrics(numeric_metrics, step=step)


def log_artifact(path: Path, artifact_path: Optional[str] = None) -> None:
    """Log a file or directory as an MLflow artifact."""

    if mlflow is None:
        return

    if path.is_dir():
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)
    else:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_json(obj: Dict[str, Any], artifact_file: str) -> None:
    """Log a JSON payload as an MLflow artifact."""

    if mlflow is None:
        return

    mlflow.log_dict(obj, artifact_file)


__all__ = [
    "_flatten",
    "_parse_tags",
    "start_tracking",
    "log_params",
    "log_metrics",
    "log_artifact",
    "log_json",
]
