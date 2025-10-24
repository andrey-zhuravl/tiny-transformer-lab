"""Lightweight utilities for optional MLflow logging."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


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

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
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


__all__ = ["mlflow_run"]
