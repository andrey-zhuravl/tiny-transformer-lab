"""Dataset processing utilities (statistics, splitting, manifest generation)."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy missing
    np = None  # type: ignore

from .validate import (
    DATA_FORMAT_JSONL,
    DATA_FORMAT_PARQUET,
    Schema,
    ValidationReport,
    _iter_records,
    validate_dataset,
)


class DatasetProcessingError(RuntimeError):
    """Raised when dataset processing fails."""


@dataclass(slots=True)
class DatasetStatistics:
    rows_total: int
    unique_ids: int
    task_distribution: Mapping[str, int]
    label_distribution: Mapping[str, Mapping[str, int]]
    field_coverage: Mapping[str, float]
    meta_field_coverage: Mapping[str, float]
    ngram_vocab: Mapping[str, int]
    text_length: Mapping[str, float]
    noise_rate: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "rows_total": self.rows_total,
            "unique_ids": self.unique_ids,
            "task_distribution": dict(self.task_distribution),
            "label_distribution": {k: dict(v) for k, v in self.label_distribution.items()},
            "field_coverage": dict(self.field_coverage),
            "meta_field_coverage": dict(self.meta_field_coverage),
            "ngram_vocab": dict(self.ngram_vocab),
            "text_length": dict(self.text_length),
            "noise_rate": self.noise_rate,
        }


@dataclass(slots=True)
class SplitArtifact:
    name: str
    path: Path
    rows: int
    sha256: str


@dataclass(slots=True)
class ProcessResult:
    manifest: Dict[str, object]
    manifest_path: Path
    stats: DatasetStatistics
    stats_path: Path
    split_artifacts: Sequence[SplitArtifact]
    validation_report: ValidationReport
    mlflow_run_id: Optional[str]
    duration_sec: float


def _normalise_splits(raw_splits: Mapping[str, float]) -> Dict[str, float]:
    if not raw_splits:
        raise DatasetProcessingError("At least one split must be specified")

    for name, value in raw_splits.items():
        if value < 0:
            raise DatasetProcessingError(f"Split '{name}' ratio must be non-negative")

    total = sum(raw_splits.values())
    if total <= 0:
        raise DatasetProcessingError("Split ratios must sum to a positive value")

    return {name: value / total for name, value in raw_splits.items()}


def _count_records(records: Sequence[MutableMapping[str, object]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        task = record.get("task")
        counter[task] += 1
    return counter


def _label_distribution(records: Sequence[MutableMapping[str, object]]) -> Dict[str, Dict[str, int]]:
    distribution: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        task = record.get("task")
        if not isinstance(task, Mapping):
            continue
        task_type = task.get("type")
        if not isinstance(task_type, str):
            continue
        label = task.get("label")
        if isinstance(label, str):
            distribution[task_type][label] += 1
    return distribution


def _field_coverage(
    records: Sequence[MutableMapping[str, object]],
    fields: Iterable[str],
) -> Dict[str, float]:
    coverage: Dict[str, float] = {}
    total = len(records)
    if total == 0:
        return {field: 0.0 for field in fields}

    for field in fields:
        count = 0
        for record in records:
            value = record.get(field)
            if isinstance(value, str):
                present = value != ""
            else:
                present = value is not None
            if present:
                count += 1
        coverage[field] = round(count / total, 6)
    return coverage


def _meta_field_coverage(
    records: Sequence[MutableMapping[str, object]], fields: Iterable[str]
) -> Dict[str, float]:
    coverage: Dict[str, float] = {}
    total = len(records)
    if total == 0:
        return {field: 0.0 for field in fields}

    for field in fields:
        count = 0
        for record in records:
            meta = record.get("meta")
            if isinstance(meta, Mapping):
                value = meta.get(field)
                if isinstance(value, str):
                    present = value != ""
                else:
                    present = value is not None
                if present:
                    count += 1
        coverage[field] = round(count / total, 6)
    return coverage


def _ngram_vocab(records: Sequence[MutableMapping[str, object]]) -> Dict[str, int]:
    vocab: Dict[str, set[str]] = defaultdict(set)
    for record in records:
        task = record.get("task")
        if not isinstance(task, Mapping):
            continue
        task_type = task.get("type")
        if not isinstance(task_type, str):
            continue
        for value in task.values():
            if isinstance(value, str):
                for token in value.split():
                    vocab[task_type].add(token)
    return {task_type: len(tokens) for task_type, tokens in vocab.items()}


def _text_length_stats(records: Sequence[MutableMapping[str, object]]) -> Dict[str, float]:
    lengths: List[int] = []
    for record in records:
        task = record.get("task")
        if not isinstance(task, Mapping):
            continue
        for value in task.values():
            if isinstance(value, str):
                lengths.append(len(value))

    if not lengths:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}

    mean = statistics.mean(lengths)
    stdev = statistics.pstdev(lengths)
    return {
        "mean": round(mean, 6),
        "stdev": round(stdev, 6),
        "min": float(min(lengths)),
        "max": float(max(lengths)),
    }


def _noise_rate(records: Sequence[MutableMapping[str, object]]) -> float:
    if not records:
        return 0.0
    noisy = 0
    for record in records:
        meta = record.get("meta")
        if isinstance(meta, Mapping) and bool(meta.get("is_noisy")):
            noisy += 1
    return round(noisy / len(records), 6)


def compute_statistics(
    records: Sequence[MutableMapping[str, object]],
    schema: Schema,
) -> DatasetStatistics:
    dataset_section = schema.get("dataset", {})
    dataset_fields_value = dataset_section.get("required_fields", [])
    dataset_fields = (
        list(dataset_fields_value)
        if isinstance(dataset_fields_value, Iterable) and not isinstance(dataset_fields_value, (str, bytes))
        else []
    )
    meta_section = dataset_section.get("meta_fields", {})
    meta_required_value = meta_section.get("required", []) if isinstance(meta_section, Mapping) else []
    meta_fields = (
        list(meta_required_value)
        if isinstance(meta_required_value, Iterable) and not isinstance(meta_required_value, (str, bytes))
        else []
    )

    rows_total = len(records)
    unique_ids = len({record.get("id") for record in records})
    task_distribution = _count_records(records)
    label_distribution = _label_distribution(records)
    field_coverage = _field_coverage(records, dataset_fields)
    meta_field_coverage = _meta_field_coverage(records, meta_fields)
    ngram_vocab = _ngram_vocab(records)
    text_length = _text_length_stats(records)
    noise_rate = _noise_rate(records)

    return DatasetStatistics(
        rows_total=rows_total,
        unique_ids=unique_ids,
        task_distribution=task_distribution,
        label_distribution=label_distribution,
        field_coverage=field_coverage,
        meta_field_coverage=meta_field_coverage,
        ngram_vocab=ngram_vocab,
        text_length=text_length,
        noise_rate=noise_rate,
    )


def _allocate_split_counts(total: int, splits: Mapping[str, float]) -> Dict[str, int]:
    raw_counts = {name: value * total for name, value in splits.items()}
    floored = {name: math.floor(count) for name, count in raw_counts.items()}
    remainder = total - sum(floored.values())
    if remainder > 0:
        fractions = sorted(
            ((raw_counts[name] - floored[name], name) for name in splits),
            reverse=True,
        )
        if fractions:
            index = 0
            while remainder > 0:
                _, name = fractions[index % len(fractions)]
                floored[name] += 1
                remainder -= 1
                index += 1
    return floored


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_split(
    name: str,
    records: Sequence[MutableMapping[str, object]],
    indices: Sequence[int],
    output_dir: Path,
) -> SplitArtifact:
    output_path = output_dir / f"{name}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for index in indices:
            json.dump(records[index], handle, sort_keys=True)
            handle.write("\n")
    return SplitArtifact(name=name, path=output_path, rows=len(indices), sha256=_sha256(output_path))


def perform_splits(
    records: Sequence[MutableMapping[str, object]],
    *,
    splits: Mapping[str, float],
    seed: int,
    output_dir: Path,
) -> List[SplitArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(records)
    if total == 0:
        return [
            SplitArtifact(name=name, path=output_dir / f"{name}.jsonl", rows=0, sha256="")
            for name in splits
        ]

    if np is not None:
        generator = np.random.default_rng(seed)
        indices = generator.permutation(total).tolist()
    else:  # pragma: no cover - exercised in environments without numpy
        rng = random.Random(seed)
        indices = list(range(total))
        rng.shuffle(indices)

    counts = _allocate_split_counts(total, splits)

    split_artifacts: List[SplitArtifact] = []
    cursor = 0
    for name, count in counts.items():
        slice_indices = indices[cursor : cursor + count]
        cursor += count
        artifact = _write_split(name, records, slice_indices, output_dir)
        split_artifacts.append(artifact)
    return split_artifacts


def _extract_grammar_rev(records: Sequence[MutableMapping[str, object]]) -> Optional[str]:
    for record in records:
        meta = record.get("meta")
        if isinstance(meta, Mapping):
            grammar_rev = meta.get("grammar_rev")
            if isinstance(grammar_rev, str):
                return grammar_rev
    return None


def process_dataset(
    *,
    dataset_path: Path,
    schema_path: Path,
    schema: Schema,
    data_format: str,
    split_ratios: Mapping[str, float],
    seed: int,
    output_dir: Path,
) -> ProcessResult:
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    if data_format not in {DATA_FORMAT_JSONL, DATA_FORMAT_PARQUET}:
        raise DatasetProcessingError(
            f"Unsupported processing format '{data_format}'. Expected JSONL or PARQUET"
        )

    validation_report = validate_dataset(dataset_path, schema, data_format)
    if validation_report.rows_invalid:
        raise DatasetProcessingError("Dataset contains invalid rows; aborting processing")

    records = list(_iter_records(dataset_path, data_format))
    normalised_splits = _normalise_splits(split_ratios)

    dataset_id = dataset_path.stem

    stats = compute_statistics(records, schema)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    split_artifacts = perform_splits(
        records,
        splits=normalised_splits,
        seed=seed,
        output_dir=output_dir,
    )
    duration = time.perf_counter() - start_time

    manifest: Dict[str, object] = {
        "dataset_id": dataset_id,
        "grammar_rev": _extract_grammar_rev(records),
        "seed": seed,
        "splits": {
            artifact.name: {
                "path": str(artifact.path),
                "rows": artifact.rows,
                "sha256": artifact.sha256,
            }
            for artifact in split_artifacts
        },
        "format": data_format,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schema_path": str(schema_path),
    }

    manifest_path = output_dir / "dataset.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    stats_path = output_dir / "dataset_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats.to_dict(), handle, indent=2, sort_keys=True)

    mlflow_run_id: Optional[str] = None

    return ProcessResult(
        manifest=manifest,
        manifest_path=manifest_path,
        stats=stats,
        stats_path=stats_path,
        split_artifacts=split_artifacts,
        validation_report=validation_report,
        mlflow_run_id=mlflow_run_id,
        duration_sec=duration,
    )


__all__ = [
    "DatasetProcessingError",
    "DatasetStatistics",
    "ProcessResult",
    "SplitArtifact",
    "compute_statistics",
    "perform_splits",
    "process_dataset",
]
