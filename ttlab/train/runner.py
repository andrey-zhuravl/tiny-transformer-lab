"""Training and evaluation routines for Tiny Transformer Lab models."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ttlab.models import BaseTTLabModel, create_model

try:  # pragma: no cover - optional dependency for telemetry
    import mlflow
except Exception:  # pragma: no cover - fallback when MLflow is not installed
    mlflow = None  # type: ignore


class JsonlDataset(Dataset):
    """Dataset backed by a JSONL file with tokenised samples."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows: List[MutableMapping[str, object]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        row = self.rows[index]
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        attention = row.get("attention_mask")
        if attention is None:
            attention = [1] * len(row["input_ids"])
        attention_mask = torch.tensor(attention, dtype=torch.long)
        sample: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        labels = row.get("labels")
        if labels is not None:
            sample["labels"] = torch.tensor(labels, dtype=torch.long)
        if "graph_edges" in row:
            sample["graph_edges"] = row["graph_edges"]
        if "roles" in row:
            sample["roles"] = row["roles"]
        return sample


def _pad_sequences(tensors: Sequence[Tensor], *, padding_value: int) -> Tensor:
    return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def collate_batch(batch: Sequence[Dict[str, Tensor]], pad_id: int = 0) -> Dict[str, Tensor]:
    input_ids = _pad_sequences([sample["input_ids"] for sample in batch], padding_value=pad_id)
    attention_masks = _pad_sequences(
        [sample["attention_mask"] for sample in batch], padding_value=0
    )
    collated: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_masks}

    label_tensors = [sample.get("labels") for sample in batch]
    if any(tensor is not None for tensor in label_tensors):
        filtered = [tensor if tensor is not None else torch.tensor([], dtype=torch.long) for tensor in label_tensors]
        collated["labels"] = _pad_sequences(filtered, padding_value=-100)

    edge_lists = [sample.get("graph_edges") for sample in batch]
    if any(edges is not None for edges in edge_lists):
        collated["graph_edges"] = [list(edges) if edges is not None else [] for edges in edge_lists]

    role_lists = [sample.get("roles") for sample in batch]
    if any(role is not None for role in role_lists):
        collated["roles"] = role_lists

    return collated


def collate(batch: Sequence[Dict[str, Tensor]], pad_id: int = 0) -> Dict[str, Tensor]:
    """Alias maintained for external modules importing ``collate``."""

    return collate_batch(batch, pad_id=pad_id)


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - GPU specific branch
        torch.cuda.manual_seed_all(seed)


def _maybe_start_mlflow_run(mlflow_uri: Optional[str]):
    if mlflow_uri is None:
        return None
    if mlflow is None:  # pragma: no cover - executed when MLflow missing
        raise RuntimeError("MLflow is not installed. Install tiny-transformer-lab[mlflow].")
    mlflow.set_tracking_uri(mlflow_uri)
    return mlflow.start_run()


def _iter_scalar_params(mapping: Mapping[str, object], prefix: str) -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    for key, value in mapping.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, Mapping):
            items.extend(_iter_scalar_params(value, f"{name}."))
        elif isinstance(value, (str, int, float, bool)):
            items.append((name, value))
    return items


def _log_mlflow_params(cfg: Mapping[str, object]) -> None:
    if mlflow is None:
        return
    model_cfg = cfg.get("model", {})
    trainer_cfg = cfg.get("trainer", {})
    tokenizer_cfg = cfg.get("tokenizer", {})
    if isinstance(model_cfg, Mapping):
        for key, value in _iter_scalar_params(model_cfg, "model."):
            mlflow.log_param(key, value)
    if isinstance(trainer_cfg, Mapping):
        for key, value in _iter_scalar_params(trainer_cfg, "trainer."):
            mlflow.log_param(key, value)
    if isinstance(tokenizer_cfg, Mapping):
        for key, value in _iter_scalar_params(tokenizer_cfg, "tokenizer."):
            mlflow.log_param(key, value)


def _log_mlflow_tags(cfg: Mapping[str, object]) -> None:
    if mlflow is None:
        return
    model_cfg = cfg.get("model", {})
    kind = "unknown"
    if isinstance(model_cfg, Mapping):
        kind = str(model_cfg.get("kind", "unknown"))
    family = kind if kind in {"hyper", "graph"} else "baseline"
    mlflow.set_tags({"family": family, "task": "lm", "struct": "generalization"})


def _log_configs(cfg: Mapping[str, object]) -> None:
    if mlflow is None:
        return
    import tempfile
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_file = tmp_path / "resolved_model.yaml"
        trainer_file = tmp_path / "resolved_trainer.yaml"
        with model_file.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg.get("model", {}), handle)
        with trainer_file.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg.get("trainer", {}), handle)
        mlflow.log_artifact(model_file, artifact_path="configs")
        mlflow.log_artifact(trainer_file, artifact_path="configs")


def _log_loss_plot(steps: Sequence[int], losses: Sequence[float]) -> None:
    if mlflow is None or not steps:
        return
    try:  # pragma: no cover - requires optional dependency
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - fallback when matplotlib missing
        return
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "loss.png"
        plt.figure(figsize=(4, 3))
        plt.plot(steps, losses, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Train loss")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        mlflow.log_artifact(path, artifact_path="plots")


def _tokens_in_batch(batch: Mapping[str, Tensor]) -> int:
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        return int(attention_mask.sum().item())
    return int(batch["input_ids"].numel())


def _apply_warmup(optimizer: torch.optim.Optimizer, step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    if step <= warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        lr = base_lr
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train(
    data_dir: str | os.PathLike[str],
    cfg: Mapping[str, object] | DictConfig,
    *,
    seed: int = 212,
    mlflow_uri: Optional[str] = None,
) -> Path:
    """Train the configured model on the dataset located in ``data_dir``."""

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg, Mapping):  # pragma: no cover - defensive
            raise TypeError("Resolved trainer config must be a mapping")

    data_path = Path(data_dir)
    train_path = data_path / "train.jsonl"
    dev_path = data_path / "dev.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    _set_seed(seed)
    trainer_cfg = cfg["trainer"]
    batch_size = int(trainer_cfg["batch_size"])
    num_workers = int(trainer_cfg.get("num_workers", 0))

    train_dataset = JsonlDataset(train_path)
    dev_dataset = JsonlDataset(dev_path) if dev_path.exists() else None

    pad_id = int(cfg.get("tokenizer", {}).get("pad_token_id", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_id=pad_id),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = _resolve_device(str(trainer_cfg.get("device", "auto")))
    vocab_size = int(cfg.get("tokenizer", {}).get("vocab_size", 32000))
    model = create_model(cfg, vocab_size=vocab_size)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer_cfg["lr"]),
        betas=tuple(trainer_cfg.get("betas", (0.9, 0.95))),
        weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
    )

    grad_clip = float(trainer_cfg.get("grad_clip", 0.0))
    warmup_steps = int(trainer_cfg.get("warmup_steps", 0))
    max_steps = int(trainer_cfg.get("max_steps", 1000))
    log_interval = int(trainer_cfg.get("log_interval", 50))
    eval_interval = int(trainer_cfg.get("eval_interval", 200))

    run = _maybe_start_mlflow_run(mlflow_uri)
    try:
        if run is not None:
            _log_mlflow_params(cfg)
            _log_mlflow_tags(cfg)
            _log_configs(cfg)
            mlflow.log_param("seed", seed)

        step = 0
        best_dev = math.inf
        rolling_tokens = 0
        rolling_time = 0.0
        history_steps: List[int] = []
        history_losses: List[float] = []
        start_time = time.perf_counter()

        while step < max_steps:
            for batch in train_loader:
                step += 1
                batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch)
                loss = model.compute_loss(batch, outputs)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                _apply_warmup(optimizer, step, float(trainer_cfg["lr"]), warmup_steps)
                optimizer.step()

                elapsed = time.perf_counter() - start_time
                tokens = _tokens_in_batch(batch)
                rolling_tokens += tokens
                rolling_time += elapsed
                start_time = time.perf_counter()

                if step % log_interval == 0:
                    train_loss = float(loss.detach().cpu().item())
                    history_steps.append(step)
                    history_losses.append(train_loss)
                    if run is not None:
                        mlflow.log_metric("train/loss", train_loss, step=step)
                        if rolling_time > 0:
                            mlflow.log_metric(
                                "speed/tokens_per_s", rolling_tokens / rolling_time, step=step
                            )
                            mlflow.log_metric(
                                "time/step_ms", (rolling_time / log_interval) * 1000.0, step=step
                            )
                        if torch.cuda.is_available() and device.type == "cuda":  # pragma: no cover
                            mlflow.log_metric(
                                "memory/peak_mb",
                                torch.cuda.max_memory_allocated(device) / (1024 * 1024),
                                step=step,
                            )
                    rolling_tokens = 0
                    rolling_time = 0.0

                if dev_dataset is not None and step % eval_interval == 0:
                    metrics = evaluate(model, dev_dataset, batch_size, device=device)
                    dev_loss = metrics["loss"]
                    best_dev = min(best_dev, dev_loss)
                    if run is not None:
                        mlflow.log_metric("dev/loss", dev_loss, step=step)
                        mlflow.log_metric("dev/ppl", metrics["ppl"], step=step)

                if step >= max_steps:
                    break

        checkpoint_dir = Path("runs")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_kind = str(cfg.get("model", {}).get("kind", "model"))
        checkpoint_path = checkpoint_dir / f"{model_kind}_step{step}.pt"
        torch.save({"model": model.state_dict(), "cfg": cfg}, checkpoint_path)

        if run is not None:
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            _log_loss_plot(history_steps, history_losses)

        return checkpoint_path
    finally:
        if run is not None:
            mlflow.end_run()


@torch.no_grad()
def evaluate(
    model: BaseTTLabModel,
    dataset: Dataset[Dict[str, Tensor]],
    batch_size: int,
    *,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the provided model on the dataset."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        outputs = model(batch)
        loss = model.compute_loss(batch, outputs)
        total_loss += float(loss.detach().cpu().item()) * _tokens_in_batch(batch)
        total_tokens += _tokens_in_batch(batch)
    model.train()
    average_loss = total_loss / max(1, total_tokens)
    return {"loss": average_loss, "ppl": math.exp(min(20.0, average_loss))}


def evaluate_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    data_dir: str | os.PathLike[str],
    cfg: Mapping[str, object] | DictConfig,
    *,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Load a checkpoint and run evaluation on ``dev.jsonl``."""

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg, Mapping):  # pragma: no cover - defensive
            raise TypeError("Resolved evaluation config must be a mapping")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    stored_cfg = checkpoint.get("cfg")
    if isinstance(stored_cfg, DictConfig):
        stored_cfg = OmegaConf.to_container(stored_cfg, resolve=True)
    if isinstance(stored_cfg, Mapping):
        merged_cfg = dict(stored_cfg)
        merged_cfg.update(cfg)
        cfg = merged_cfg
    trainer_cfg = cfg["trainer"]
    batch_size = int(trainer_cfg["batch_size"])
    device_obj = _resolve_device(device or str(trainer_cfg.get("device", "auto")))
    vocab_size = int(cfg.get("tokenizer", {}).get("vocab_size", 32000))
    model = create_model(cfg, vocab_size=vocab_size)
    model.load_state_dict(checkpoint["model"])
    model.to(device_obj)
    dev_dataset = JsonlDataset(Path(data_dir) / "dev.jsonl")
    return evaluate(model, dev_dataset, batch_size, device=device_obj)


@torch.no_grad()
def evaluate_ckpt(
    checkpoint_path: str | os.PathLike[str],
    data_dir: str | os.PathLike[str],
    *,
    device: str | None = None,
) -> tuple[float, float]:
    """Evaluate a checkpoint using the configuration stored in the file."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("cfg")
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg, Mapping):  # pragma: no cover - defensive
        raise ValueError("Checkpoint does not contain a configuration for evaluation")
    metrics = evaluate_checkpoint(checkpoint_path, data_dir, cfg, device=device)
    return metrics["loss"], metrics["ppl"]
