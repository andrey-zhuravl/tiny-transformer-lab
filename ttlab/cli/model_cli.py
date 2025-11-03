"""Typer commands for interacting with TTLab models."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import typer

from ttlab.utils.paths import get_project_path

try:  # pragma: no cover - optional dependency handling
    from hydra import compose, initialize_config_dir  # type: ignore
except ImportError as exc:  # pragma: no cover - executed when hydra missing
    compose = None  # type: ignore
    initialize_config_dir = None  # type: ignore
    _HYDRA_ERROR = exc
else:  # pragma: no cover - trivial branch
    _HYDRA_ERROR = None

try:  # pragma: no cover - optional dependency handling
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ImportError as exc:  # pragma: no cover - executed when omegaconf missing
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore
    _OMEGACONF_ERROR = exc
else:  # pragma: no cover - trivial branch
    _OMEGACONF_ERROR = None


model_app = typer.Typer(help="Model management commands")


def _require_hydra() -> None:
    if compose is None or initialize_config_dir is None:
        raise RuntimeError(
            "Hydra is required for model commands. Install tiny-transformer-lab[models]."
        ) from _HYDRA_ERROR


def _require_omegaconf() -> None:
    if OmegaConf is None:
        raise RuntimeError(
            "omegaconf is required for model commands. Install tiny-transformer-lab[models]."
        ) from _OMEGACONF_ERROR


def _compose(overrides: List[str]) -> Any:
    _require_hydra()
    _require_omegaconf()
    config_dir = str(get_project_path("conf"))
    with initialize_config_dir(version_base=None, config_dir=config_dir, job_name="ttlab-model"):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _to_dict(cfg: Any) -> Dict[str, object]:
    _require_omegaconf()
    container = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[operator]
    if not isinstance(container, dict):  # pragma: no cover - defensive
        raise TypeError("Resolved Hydra config must be a mapping")
    return container


def _import_models():
    try:
        from ttlab import models
    except ModuleNotFoundError as exc:  # pragma: no cover - executed when torch missing
        raise RuntimeError("PyTorch is required for model commands. Install tiny-transformer-lab with the 'models' extra.") from exc
    return models


def _import_runner():
    try:
        from ttlab.train import runner
    except ModuleNotFoundError as exc:  # pragma: no cover - executed when torch missing
        raise RuntimeError("PyTorch is required for model commands. Install tiny-transformer-lab with the 'models' extra.") from exc
    return runner


@model_app.command("list")
def list_cmd() -> None:
    """Print the registered model names."""

    models = _import_models()
    for name in models.list_models():
        typer.echo(name)


@model_app.command("info")
def info_cmd(
    model: str = typer.Option("model=vanilla", "--model", help="Hydra override, e.g. model=vanilla"),
    trainer: str = typer.Option(
        "trainer=default", "--trainer", help="Trainer Hydra override (used for device/batch)"
    ),
) -> None:
    """Show the resolved config and parameter count for a model."""

    cfg = _compose([model, trainer])
    resolved = _to_dict(cfg)
    vocab_size = int(resolved.get("tokenizer", {}).get("vocab_size", 32000))
    models = _import_models()
    instance = models.create_model(resolved, vocab_size=vocab_size)
    typer.echo(OmegaConf.to_yaml(cfg))
    typer.echo(f"Parameter count: {instance.param_count():,}")


@model_app.command("train")
def train_cmd(
    data_dir: str = typer.Option(..., "--data-dir", help="Directory with train.jsonl/dev.jsonl"),
    model: str = typer.Option("model=vanilla", "--model", help="Hydra override for the model"),
    trainer: str = typer.Option(
        "trainer=default", "--trainer", help="Hydra override for the trainer configuration"
    ),
    mlflow_uri: str | None = typer.Option(
        None, "--mlflow-uri", help="Optional MLflow tracking URI for logging"
    ),
    seed: int = typer.Option(212, "--seed", help="Random seed"),
) -> None:
    """Train a model using the provided dataset and configuration."""

    cfg = _compose([model, trainer])
    resolved = _to_dict(cfg)
    runner = _import_runner()
    checkpoint = runner.train(data_dir=data_dir, cfg=resolved, seed=seed, mlflow_uri=mlflow_uri)
    typer.echo(str(checkpoint))


@model_app.command("eval")
def eval_cmd(
    run_path: str = typer.Option(..., "--run-path", help="Path to a saved checkpoint"),
    data_dir: str = typer.Option(..., "--data-dir", help="Directory containing dev.jsonl"),
    model: str = typer.Option("model=vanilla", "--model", help="Hydra override for the model"),
    trainer: str = typer.Option(
        "trainer=default", "--trainer", help="Hydra override for the trainer configuration"
    ),
    device: str | None = typer.Option(None, "--device", help="Override evaluation device"),
) -> None:
    """Evaluate a checkpoint on the dev split and print loss / perplexity."""

    cfg = _compose([model, trainer])
    resolved = _to_dict(cfg)
    runner = _import_runner()
    metrics = runner.evaluate_checkpoint(run_path, data_dir, resolved, device=device)
    typer.echo(json.dumps(metrics, indent=2))
