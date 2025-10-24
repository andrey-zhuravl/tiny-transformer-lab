from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from ..utils.files import ensure_dir, read_json, write_json

REGISTRY_PATH = Path(__file__).resolve().parent / "index.json"


def _load_registry() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        data = read_json(REGISTRY_PATH)
        if isinstance(data, dict):
            data.setdefault("tokenizers", [])
            return data
    return {"tokenizers": []}


def update_registry(manifest: Mapping[str, Any], manifest_path: Path) -> Path:
    """Update the local registry with ``manifest`` and return the registry path."""

    ensure_dir(REGISTRY_PATH.parent)
    registry = _load_registry()
    entries: List[Dict[str, Any]] = []
    tokenizer_id = manifest.get("tokenizer_id")
    for entry in registry.get("tokenizers", []):
        if entry.get("tokenizer_id") != tokenizer_id:
            entries.append(entry)
    entries.append(
        {
            "tokenizer_id": tokenizer_id,
            "manifest_path": str(Path(manifest_path).resolve()),
            "algo": manifest.get("algo"),
            "vocab_size": manifest.get("vocab_size"),
            "created_at": manifest.get("created_at"),
        }
    )
    entries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    registry["tokenizers"] = entries
    write_json(REGISTRY_PATH, registry)
    return REGISTRY_PATH
