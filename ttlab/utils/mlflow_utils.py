"""MLflow helper utilities used across TTL modules."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

EXPERIMENT_NAME = "ttl-tokenizers"


@dataclass(slots=True)
class _NoOpRun:
    """Fallback implementation used when MLflow is unavailable."""

    run_id: Optional[str] = None

    def log_metrics(self, metrics: Mapping[str, float]) -> None:  # pragma: no cover - trivial
        return

    def log_params(self, params: Mapping[str, Any]) -> None:  # pragma: no cover - trivial
        return

    def set_tags(self, tags: Mapping[str, Any]) -> None:  # pragma: no cover - trivial
        return

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:  # pragma: no cover - trivial
        return


@dataclass(slots=True)
class _MLflowFacade:
    client: Any
    run: Any

    @property
    def run_id(self) -> str:
        return self.run.info.run_id

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self.client.log_metrics(metrics)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self.client.log_params(params)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        self.client.set_tags(tags)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        self.client.log_artifact(path, artifact_path=artifact_path)


@contextmanager
def mlflow_run(
    enabled: bool,
    *,
    run_name: Optional[str] = None,
    tags: Optional[Mapping[str, Any]] = None,
) -> Iterable[object]:
    """Context manager that yields an MLflow run facade or a no-op implementation."""

    if not enabled:
        yield _NoOpRun()
        return

    try:  # pragma: no cover - exercised only when MLflow is installed
        import mlflow
    except Exception:  # pragma: no cover - fallback path when MLflow unavailable
        yield _NoOpRun()
        return

    active_run = mlflow.start_run(run_name=run_name)
    try:
        if tags:
            mlflow.set_tags(dict(tags))
        facade = _MLflowFacade(client=mlflow, run=active_run)
        yield facade
    finally:
        mlflow.end_run()


def log_tokenizer_run(
    *,
    run_name: str,
    stats: Optional[Mapping[str, float]] = None,
    params: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Log metrics, params and artifacts to MLflow when the package is available."""

    try:  # Import lazily so that MLflow remains optional in local setups.
        import mlflow  # type: ignore
    except Exception:
        return None

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as active_run:  # type: ignore[attr-defined]
        if stats:
            for key, value in stats.items():
                try:
                    mlflow.log_metric(key, float(value))
                except Exception:
                    continue
        if params:
            for key, value in params.items():
                try:
                    mlflow.log_param(key, value)
                except Exception:
                    continue
        if artifacts:
            for name, payload in artifacts.items():
                artifact_path = Path("tokenizer")
                try:
                    if isinstance(payload, (str, Path)):
                        mlflow.log_artifact(str(payload), artifact_path=str(artifact_path))
                    else:
                        mlflow.log_dict(payload, str(artifact_path / f"{name}.json"))  # type: ignore[attr-defined]
                except Exception:
                    continue
        return active_run.info.run_id  # type: ignore[attr-defined]


__all__ = ["mlflow_run", "log_tokenizer_run", "EXPERIMENT_NAME"]
