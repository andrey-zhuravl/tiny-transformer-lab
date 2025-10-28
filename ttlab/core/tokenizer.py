from __future__ import annotations

import random
import shutil
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional

from ..registry import update_registry
from ..utils.files import ensure_dir, iter_jsonl_texts, read_json, sha256_file, sha256_map, write_json
from .tok_report import generate_report

SPECIAL_TOKENS: List[str] = ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>"]
SUPPORTED_ALGOS = {"char", "bpe", "unigram"}
SUPPORTED_NORMS = {"none", "nfc"}
SUPPORTED_PUNCT_POLICIES = {"keep", "strip", "space"}

try:  # Optional import for environments without HuggingFace tokenizers
    from tokenizers import Tokenizer, Regex, decoders, models, normalizers, pre_tokenizers, processors, trainers
except Exception:  # pragma: no cover - handled at runtime
    Tokenizer = None  # type: ignore[misc,assignment]
    Regex = None  # type: ignore[misc,assignment]
    decoders = None  # type: ignore[misc,assignment]
    models = None  # type: ignore[misc,assignment]
    normalizers = None  # type: ignore[misc,assignment]
    pre_tokenizers = None  # type: ignore[misc,assignment]
    processors = None  # type: ignore[misc,assignment]
    trainers = None  # type: ignore[misc,assignment]


@dataclass(slots=True)
class TokConfig:
    algo: str
    vocab_size: int
    norm: str
    lower: bool
    punct_policy: str
    seed: int

    def manifest_fragment(self) -> Dict[str, object]:
        return {
            "algo": self.algo,
            "vocab_size": self.vocab_size,
            "normalizer": {"nfc": self.norm == "nfc", "lower": self.lower, "punct_policy": self.punct_policy},
            "seed": self.seed,
            "special_tokens": SPECIAL_TOKENS,
        }


def _validate_args(config: TokConfig) -> None:
    if config.algo not in SUPPORTED_ALGOS:
        raise ValueError(f"Unsupported tokenizer algorithm: {config.algo}")
    if config.norm not in SUPPORTED_NORMS:
        raise ValueError(f"Unsupported normalization mode: {config.norm}")
    if config.punct_policy not in SUPPORTED_PUNCT_POLICIES:
        raise ValueError(f"Unsupported punctuation policy: {config.punct_policy}")
    if config.vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError("vocab_size must exceed the number of special tokens")


def _python_normalize(text: str, config: TokConfig) -> str:
    if config.norm == "nfc":
        text = unicodedata.normalize("NFC", text)
    if config.lower:
        text = text.lower()
    if config.punct_policy != "keep":
        chars: List[str] = []
        for char in text:
            if unicodedata.category(char).startswith("P"):
                if config.punct_policy == "space":
                    chars.append(" ")
                # ``strip`` omits the punctuation character entirely.
            else:
                chars.append(char)
        text = "".join(chars)
        text = " ".join(text.split())
    return text


def _build_normalizer(config: TokConfig):
    if normalizers is None or Regex is None:
        if config.norm == "none" and not config.lower and config.punct_policy == "keep":
            return None
        raise RuntimeError("The HuggingFace `tokenizers` package is required for normalization support.")
    steps: List[object] = []
    if config.norm == "nfc":
        steps.append(normalizers.NFC())
    if config.lower:
        steps.append(normalizers.Lowercase())
    if config.punct_policy == "strip":
        steps.append(normalizers.Replace(Regex(r"\p{P}+"), ""))
    elif config.punct_policy == "space":
        steps.append(normalizers.Replace(Regex(r"\p{P}+"), " "))
    if steps:
        steps.append(normalizers.Replace(Regex(r"\s+"), " "))
        steps.append(normalizers.Strip())
        return normalizers.Sequence(steps)
    return None


def _trainer_with_kwargs(trainer_cls, **kwargs):  # type: ignore[no-untyped-def]
    try:
        return trainer_cls(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return trainer_cls(**kwargs)


def _trainer_and_model(config: TokConfig):
    if models is None or trainers is None:
        raise RuntimeError("The HuggingFace `tokenizers` package is required for training.")
    if config.algo == "bpe":
        model = models.BPE(unk_token="<unk>")
        trainer = _trainer_with_kwargs(
            trainers.BpeTrainer,
            vocab_size=config.vocab_size,
            min_frequency=1,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
            seed=config.seed,
        )
    elif config.algo == "unigram":
        model = models.Unigram()
        trainer = _trainer_with_kwargs(
            trainers.UnigramTrainer,
            vocab_size=config.vocab_size,
            special_tokens=SPECIAL_TOKENS,
            unk_token="<unk>",
            max_piece_length=16,
            seed=config.seed,
        )
    else:  # char
        model = models.Unigram()
        trainer = _trainer_with_kwargs(
            trainers.UnigramTrainer,
            vocab_size=config.vocab_size,
            special_tokens=SPECIAL_TOKENS,
            unk_token="<unk>",
            max_piece_length=1,
            seed=config.seed,
        )
    return model, trainer


def _pre_tokenizer_for_algo(algo: str):
    if pre_tokenizers is None:
        raise RuntimeError("The HuggingFace `tokenizers` package is required for training.")
    if algo == "char":
        return pre_tokenizers.ByteLevel(add_prefix_space=False)
    return pre_tokenizers.Whitespace()


def _collect_training_files(dataset_manifest: Path) -> List[Path]:
    manifest = read_json(dataset_manifest)
    splits = manifest.get("splits", {})
    train_entry = splits.get("train")
    if not train_entry:
        raise ValueError("dataset.manifest.json must contain a train split")
    paths: List[Path] = []
    if isinstance(train_entry, Mapping):
        train_path = Path(train_entry.get("path", "")).expanduser().resolve()
        if not train_path.exists():
            raise FileNotFoundError(f"Train split not found: {train_path}")
        paths.append(train_path)
    elif isinstance(train_entry, list):
        for item in train_entry:
            train_path = Path(item.get("path", "")).expanduser().resolve()
            if not train_path.exists():
                raise FileNotFoundError(f"Train split not found: {train_path}")
            paths.append(train_path)
    else:
        raise ValueError("train split must be an object or list")
    return paths


def _iter_training_texts(files: Iterable[Path], config: TokConfig) -> Iterator[str]:
    for file_path in files:
        for text in iter_jsonl_texts(file_path):
            yield _python_normalize(text, config)


def _ensure_special_tokens_present(tokenizer) -> Dict[str, int]:  # type: ignore[no-untyped-def]
    """Ensure that ``tokenizer`` contains all SPECIAL_TOKENS and return their ids."""

    missing: List[str] = []
    for token in SPECIAL_TOKENS:
        if tokenizer.token_to_id(token) is None:
            missing.append(token)
    if missing:
        tokenizer.add_special_tokens(missing)

    special_token_ids: Dict[str, int] = {}
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise RuntimeError(f"Failed to register special token '{token}' with the tokenizer backend.")
        special_token_ids[token] = token_id
    return special_token_ids


def _prepare_tokenizer(config: TokConfig, dataset_manifest: Path, out_dir: Path):
    if Tokenizer is None:
        raise RuntimeError("The HuggingFace `tokenizers` package is required for tokenizer training.")

    ensure_dir(out_dir)
    model, trainer = _trainer_and_model(config)
    tokenizer = Tokenizer(model)
    normalizer = _build_normalizer(config)
    if normalizer is not None:
        tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = _pre_tokenizer_for_algo(config.algo)
    if decoders is not None and config.algo == "char":
        tokenizer.decoder = decoders.ByteLevel()

    files = _collect_training_files(dataset_manifest)
    iterator = _iter_training_texts(files, config)
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    special_token_ids = _ensure_special_tokens_present(tokenizer)

    pad_id = special_token_ids["<pad>"]
    tokenizer.enable_padding(direction="right", pad_id=pad_id, pad_token="<pad>")

    bos_id = special_token_ids["<bos>"]
    eos_id = special_token_ids["<eos>"]
    sep_id = special_token_ids["<sep>"]
    if bos_id is not None and eos_id is not None and processors is not None:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<bos> $A <eos>",
            pair="<bos> $A <sep> $B <eos>",
            special_tokens=[
                ("<bos>", bos_id),
                ("<eos>", eos_id),
                ("<sep>", sep_id if sep_id is not None else eos_id),
            ],
        )

    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    return tokenizer, tokenizer_path, files


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]).decode().strip()
    except Exception:
        return "unknown"


def _tokenizer_id(config: TokConfig) -> str:
    norm_fragment = config.norm
    lower_fragment = "lower" if config.lower else "nolower"
    punct_fragment = config.punct_policy
    return f"{config.algo}-{config.vocab_size}-{norm_fragment}-{lower_fragment}-{punct_fragment}-s{config.seed}"


def train_or_import_tokenizer(
    *,
    dataset_manifest: Path,
    algo: str,
    vocab_size: int,
    norm: str,
    lower: bool,
    punct_policy: str,
    seed: int,
    out_dir: Path,
    external_tokenizer: Optional[Path] = None,
) -> Dict[str, object]:
    config = TokConfig(algo=algo, vocab_size=vocab_size, norm=norm, lower=lower, punct_policy=punct_policy, seed=seed)
    _validate_args(config)

    ensure_dir(out_dir)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - numpy optional
        pass

    tokenizer = None
    tokenizer_path: Optional[Path] = None
    training_files: List[Path] = []
    training_time: Optional[float] = None

    if external_tokenizer is not None:
        source_path = Path(external_tokenizer).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"External tokenizer not found: {source_path}")
        tokenizer_path = out_dir / "tokenizer.json"
        shutil.copy2(source_path, tokenizer_path)
        if Tokenizer is None:
            raise RuntimeError("The HuggingFace `tokenizers` package is required to import tokenizers.")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        _ensure_special_tokens_present(tokenizer)
        tokenizer.save(str(tokenizer_path))
        training_time = 0.0
    else:
        start = time.perf_counter()
        tokenizer, tokenizer_path, training_files = _prepare_tokenizer(config, dataset_manifest, out_dir)
        training_time = time.perf_counter() - start

    tokenizer_sha = sha256_file(tokenizer_path)

    manifest = config.manifest_fragment()
    manifest.update(
        {
            "tokenizer_id": _tokenizer_id(config),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tokenizer_json_path": str(tokenizer_path),
            "tokenizer_json_sha256": tokenizer_sha,
            "train_time_sec": training_time,
        }
    )

    dataset_info = read_json(dataset_manifest)
    if not training_files:
        training_files = _collect_training_files(dataset_manifest)

    train_entry = dataset_info.get("splits", {}).get("train")
    if training_files:
        train_split_path = str(training_files[0])
    elif isinstance(train_entry, Mapping):
        train_split_path = str(Path(train_entry.get("path", "")).expanduser().resolve())
    elif isinstance(train_entry, list) and train_entry:
        train_split_path = str(Path(train_entry[0].get("path", "")).expanduser().resolve())
    else:
        train_split_path = None

    manifest.update(
        {
            "source_dataset_id": dataset_info.get("dataset_id", "unknown"),
            "source_manifest_path": str(Path(dataset_manifest).resolve()),
            "train_split_path": train_split_path,
            "input_paths_sha256": sha256_map({path.name: path for path in training_files}),
            "source_git_sha": _git_sha(),
        }
    )

    manifest_path = out_dir / "tokenizer.manifest.json"
    write_json(manifest_path, manifest)

    report = generate_report(tokenizer=tokenizer, dataset_manifest=dataset_manifest, train_time_sec=training_time)
    report_path = out_dir / "tokenizer.report.json"
    write_json(report_path, report)

    registry_path = update_registry(manifest, manifest_path)

    return {
        "tokenizer_path": str(tokenizer_path),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "manifest": manifest,
        "report": report,
        "mlflow_run_id": None,
        "registry_path": str(registry_path),
    }


def inspect_tokenizer(*, tokenizer_path: Path, dataset_manifest: Path, out_dir: Path) -> Dict[str, object]:
    ensure_dir(out_dir)
    if Tokenizer is None:
        raise RuntimeError("The HuggingFace `tokenizers` package is required to inspect tokenizers.")
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    report = generate_report(tokenizer=tokenizer, dataset_manifest=dataset_manifest, train_time_sec=None)
    report_path = Path(out_dir) / "tokenizer.report.json"
    write_json(report_path, report)

    return {"report_path": str(report_path), "report": report}
