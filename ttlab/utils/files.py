from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping


def ensure_dir(path: Path | str) -> Path:
    """Create *path* (if needed) and return it as a :class:`Path`."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: Path | str) -> str:
    """Return the SHA256 checksum for ``path``."""

    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path | str, payload: Any) -> None:
    """Serialise ``payload`` to ``path`` using UTF-8 encoded JSON."""

    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_json(path: Path | str) -> Any:
    """Load a JSON document from ``path``."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl_texts(path: Path | str) -> Iterator[str]:
    """Yield text-like fields from a JSONL dataset.

    The helper understands the canonical TTL dataset schema which may expose
    ``text`` (LM/CLS) or ``src``/``tgt`` (seq2seq) payloads. Empty lines and
    malformed JSON objects are ignored.
    """

    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" in record:
                yield str(record["text"])
            if "src" in record:
                yield str(record["src"])
            if "tgt" in record:
                yield str(record["tgt"])


def sha256_map(paths: Mapping[str, Path | str]) -> Dict[str, str]:
    """Return a mapping of name → sha256 for ``paths``."""

    return {name: sha256_file(path) for name, path in paths.items()}
