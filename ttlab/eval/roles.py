"""Role-wise accuracy evaluation for TTLab models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ttlab.models import create_model
from ttlab.train.runner import collate

try:  # pragma: no cover - optional dependency handling
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except Exception:  # pragma: no cover - executed when omegaconf missing
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore

try:  # pragma: no cover - optional dependency handling
    import mlflow
except Exception:  # pragma: no cover - mlflow optional
    mlflow = None  # type: ignore


class _RolesDataset(Dataset):
    """Minimal dataset that loads tokenised samples with optional role metadata."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.rows: List[Mapping[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        attention = row.get("attention_mask")
        if attention is None:
            attention = [1] * len(row["input_ids"])
        sample: Dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
        if "graph_edges" in row:
            sample["graph_edges"] = row["graph_edges"]
        if "roles" in row:
            sample["roles"] = row["roles"]
        return sample


def _normalise_cfg(cfg: object) -> Mapping[str, object]:
    if isinstance(cfg, DictConfig) and OmegaConf is not None:  # pragma: no cover - optional
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg, Mapping):
        raise ValueError("Checkpoint does not include a valid configuration")
    return cfg


def _select_device(cfg: Mapping[str, object], override: Optional[str]) -> torch.device:
    if override is not None:
        return torch.device(override)
    trainer_cfg = cfg.get("trainer", {})
    device_value = "auto"
    if isinstance(trainer_cfg, Mapping):
        device_value = str(trainer_cfg.get("device", "auto"))
    if device_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def _batch_size(cfg: Mapping[str, object]) -> int:
    trainer_cfg = cfg.get("trainer", {})
    if isinstance(trainer_cfg, Mapping):
        value = trainer_cfg.get("batch_size")
        if isinstance(value, int) and value > 0:
            return value
    return 8


def _pad_id(cfg: Mapping[str, object]) -> int:
    tokenizer_cfg = cfg.get("tokenizer", {})
    if isinstance(tokenizer_cfg, Mapping):
        value = tokenizer_cfg.get("pad_token_id")
        if isinstance(value, int):
            return value
    return 0


def _vocab_size(cfg: Mapping[str, object]) -> int:
    tokenizer_cfg = cfg.get("tokenizer", {})
    if isinstance(tokenizer_cfg, Mapping):
        value = tokenizer_cfg.get("vocab_size")
        if isinstance(value, int):
            return value
    return 32000


def _log_roles_metrics(metrics: Mapping[str, Optional[float]]) -> None:
    if mlflow is None:  # pragma: no cover - no mlflow
        return
    active = mlflow.active_run()
    if active is None:  # pragma: no cover - mlflow not active
        return
    payload = {f"roles/acc_{key}": value for key, value in metrics.items() if value is not None}
    if payload:
        mlflow.log_metrics(payload)
        mlflow.log_dict(metrics, "eval/roles.json")


@torch.no_grad()
def eval_roles_ckpt(
    ckpt_path: str | Path,
    data_dir: str | Path,
    *,
    device: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """Evaluate a checkpoint on role-specific positions if available."""

    checkpoint = torch.load(Path(ckpt_path), map_location="cpu")
    cfg = _normalise_cfg(checkpoint.get("cfg"))
    vocab_size = _vocab_size(cfg)
    model = create_model(cfg, vocab_size=vocab_size)
    state = checkpoint.get("model")
    if isinstance(state, Mapping):
        model.load_state_dict(state)  # type: ignore[arg-type]
    model_device = _select_device(cfg, device)
    model.to(model_device)
    model.eval()

    data_path = Path(data_dir) / "dev.jsonl"
    dataset = _RolesDataset(data_path)
    batch_size = _batch_size(cfg)
    pad_id = _pad_id(cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, pad_id=pad_id),
    )

    numerators = {"who": 0, "what": 0, "where": 0}
    denominators = {"who": 0, "what": 0, "where": 0}
    any_roles = False

    for batch in loader:
        model_batch: Dict[str, object] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                model_batch[key] = value.to(model_device)
            else:
                model_batch[key] = value
        outputs = model(model_batch)  # type: ignore[arg-type]
        logits = outputs["logits"]
        predictions = logits.argmax(dim=-1)
        labels = model_batch["input_ids"]
        if not isinstance(labels, Tensor):  # pragma: no cover - defensive
            continue
        roles_list = batch.get("roles")
        if roles_list is None:
            continue
        for sample_index, roles in enumerate(roles_list):
            if not isinstance(roles, Mapping):
                continue
            any_roles = True
            for key in ("who", "what", "where"):
                indices = roles.get(key)
                if not isinstance(indices, Iterable):
                    continue
                for position in indices:
                    idx = int(position)
                    if 0 <= idx < labels.size(1):
                        denominators[key] += 1
                        if predictions[sample_index, idx].item() == labels[sample_index, idx].item():
                            numerators[key] += 1

    if not any_roles:
        return {}

    metrics = {
        key: (numerators[key] / denominators[key] if denominators[key] > 0 else None)
        for key in ("who", "what", "where")
    }
    _log_roles_metrics(metrics)
    return metrics
