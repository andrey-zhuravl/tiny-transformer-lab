"""Typer commands for interacting with TTLab models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import typer

from ttlab.utils.paths import get_project_path
from ttlab.train.bench import run as bench_run
from ttlab.eval.roles import eval_roles_ckpt

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


@model_app.command("bench")
def bench_cmd(
    models: str = typer.Option("vanilla,linear,sparse", "--models", help="Comma separated model kinds"),
    seq_lens: str = typer.Option("512,2048", "--seq-lens", help="Comma separated sequence lengths"),
    d_model: int = typer.Option(128, "--d-model", help="Model hidden size"),
    n_layers: int = typer.Option(2, "--n-layers", help="Number of Transformer blocks"),
    n_heads: int = typer.Option(4, "--n-heads", help="Number of attention heads"),
    trials: int = typer.Option(3, "--trials", help="Number of timed trials"),
    device: str | None = typer.Option(None, "--device", help="Device override"),
    mlflow_uri: str | None = typer.Option(None, "--mlflow-uri", help="Optional MLflow tracking URI"),
) -> None:
    """Benchmark speed and memory for the selected models."""

    names = [name.strip() for name in models.split(",") if name.strip()]
    lengths = [int(value) for value in seq_lens.split(",") if value.strip()]
    results = bench_run(
        names,
        lengths,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        trials=trials,
        device=device,
        mlflow_uri=mlflow_uri,
    )
    typer.echo(json.dumps(results, indent=2))
    output_dir = Path("bench")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


@model_app.command("eval")
def eval_cmd(
    ckpt_path: str = typer.Option(..., "--ckpt-path", help="Path to a saved checkpoint"),
    data_dir: str = typer.Option(..., "--data-dir", help="Directory containing dev.jsonl"),
    device: str | None = typer.Option(None, "--device", help="Override evaluation device"),
) -> None:
    """Evaluate a checkpoint on the dev split and print loss / perplexity."""

    runner = _import_runner()
    loss, ppl = runner.evaluate_ckpt(ckpt_path, data_dir, device=device)
    typer.echo(json.dumps({"dev/loss": loss, "dev/ppl": ppl}, indent=2))


@model_app.command("eval-roles")
def eval_roles_cmd(
    ckpt_path: str = typer.Option(..., "--ckpt-path", help="Path to a saved checkpoint"),
    data_dir: str = typer.Option(..., "--data-dir", help="Directory containing dev.jsonl"),
    device: str | None = typer.Option(None, "--device", help="Optional evaluation device override"),
) -> None:
    """Compute role-wise accuracies for checkpoints that include role annotations."""

    metrics = eval_roles_ckpt(ckpt_path, data_dir, device=device)
    if not metrics:
        typer.echo("No role annotations found in dataset; nothing to evaluate.")
        return
    output_dir = Path("eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "roles.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(json.dumps(metrics, indent=2))

