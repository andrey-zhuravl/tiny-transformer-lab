"""Helpers for working with GardenKeeper manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class GardenManifest:
    name: str
    slug: str
    kind: str
    version: str
    tags: list[str]
    paths: Dict[str, str]
    repo: Dict[str, str]


def load_manifest(path: Path | str) -> GardenManifest:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return GardenManifest(**data)
