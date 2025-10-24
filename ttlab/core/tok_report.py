from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from ..utils.files import read_json

try:  # ``tokenizers`` is optional during import but required for metric computation.
    from tokenizers import Tokenizer
except Exception:  # pragma: no cover - handled at runtime when metrics run
    Tokenizer = Any  # type: ignore[misc,assignment]


def _percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    if not 0 <= q <= 1:
        raise ValueError("percentile quantile must be within [0, 1]")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = index - lower
    return float(lower_value + (upper_value - lower_value) * weight)


def _iter_text_samples(path: Path) -> Iterable[Dict[str, str]]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            task = str(record.get("task", "unknown"))
            if "text" in record:
                yield {"task": task, "payload": str(record["text"])}
            if "src" in record:
                yield {"task": task, "payload": str(record["src"])}
            if "tgt" in record:
                yield {"task": task, "payload": str(record["tgt"])}


def _aggregate_metrics(lengths: List[int], unk_tokens: int, total_tokens: int, total_samples: int) -> Dict[str, float]:
    oov_rate = (unk_tokens / total_tokens) if total_tokens else 0.0
    avg_len = (total_tokens / total_samples) if total_samples else 0.0
    return {
        "oov_rate": oov_rate,
        "avg_tokens_per_sample": avg_len,
        "tokens_len_p50": _percentile(lengths, 0.5),
        "tokens_len_p95": _percentile(lengths, 0.95),
    }


def generate_report(
    *,
    tokenizer: Tokenizer,
    dataset_manifest: Path,
    splits: Optional[Iterable[str]] = None,
    train_time_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute tokenization metrics for the provided dataset manifest."""

    if tokenizer is None:
        raise RuntimeError("The HuggingFace `tokenizers` package is required to compute metrics.")

    manifest = read_json(dataset_manifest)
    manifest_splits: Mapping[str, Mapping[str, Any]] = manifest.get("splits", {})
    if splits is None:
        requested_splits = list(manifest_splits.keys())
    else:
        requested_splits = list(splits)

    lengths_all: List[int] = []
    total_tokens_all = 0
    total_samples_all = 0
    total_unk_all = 0
    per_task_totals: MutableMapping[str, Dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "unk": 0.0, "samples": 0.0})
    per_split: Dict[str, Any] = {}

    unk_token_id = tokenizer.token_to_id("<unk>")

    for split_name in requested_splits:
        split_entry = manifest_splits.get(split_name)
        if not split_entry:
            continue
        split_path = Path(split_entry.get("path", "")).expanduser().resolve()
        if not split_path.exists():
            continue

        split_lengths: List[int] = []
        split_tokens = 0
        split_samples = 0
        split_unk = 0
        split_task_totals: MutableMapping[str, Dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "unk": 0.0, "samples": 0.0})

        for sample in _iter_text_samples(split_path):
            text = sample["payload"]
            task = sample["task"]
            encoding = tokenizer.encode(text, add_special_tokens=False)
            token_ids = list(encoding.ids)
            length = len(token_ids)
            split_lengths.append(length)
            split_tokens += length
            split_samples += 1
            unk_count = sum(1 for token_id in token_ids if token_id == unk_token_id)
            split_unk += unk_count
            task_bucket = split_task_totals[task]
            task_bucket["tokens"] += length
            task_bucket["unk"] += unk_count
            task_bucket["samples"] += 1

        per_split_metrics = _aggregate_metrics(split_lengths, split_unk, split_tokens, split_samples)
        per_split_metrics.update(
            {
                "num_samples": split_samples,
                "num_tokens": split_tokens,
                "coverage_per_task": {
                    task: (1 - (bucket["unk"] / bucket["tokens"])) if bucket["tokens"] else 0.0
                    for task, bucket in split_task_totals.items()
                },
            }
        )
        per_split[split_name] = per_split_metrics

        lengths_all.extend(split_lengths)
        total_tokens_all += split_tokens
        total_samples_all += split_samples
        total_unk_all += split_unk
        for task, bucket in split_task_totals.items():
            agg_bucket = per_task_totals[task]
            agg_bucket["tokens"] += bucket["tokens"]
            agg_bucket["unk"] += bucket["unk"]
            agg_bucket["samples"] += bucket["samples"]

    summary = _aggregate_metrics(lengths_all, total_unk_all, total_tokens_all, total_samples_all)
    summary.update(
        {
            "num_samples": total_samples_all,
            "num_tokens": total_tokens_all,
            "coverage_per_task": {
                task: (1 - (bucket["unk"] / bucket["tokens"])) if bucket["tokens"] else 0.0
                for task, bucket in per_task_totals.items()
            },
            "generated_at": dt.datetime.utcnow().replace(microsecond=False).isoformat() + "Z",
        }
    )
    if train_time_sec is not None:
        summary["train_time_sec"] = float(train_time_sec)

    report = {
        "summary": summary,
        "per_split": per_split,
    }
    return report
