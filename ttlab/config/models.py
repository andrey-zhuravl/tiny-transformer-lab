"""Pydantic models for ttlab configuration files."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetFormat(str, Enum):
    """Supported dataset formats."""

    JSONL = "JSONL"
    PARQUET = "PARQUET"


class DatasetField(BaseModel):
    """Describes a field in a dataset schema."""

    name: str = Field(..., alias="field_name")
    type: str
    required: bool = True

    model_config = {
        "populate_by_name": True,
        "extra": "forbid",
    }


class GeneralConfig(BaseModel):
    """General project settings."""

    seed: int = 42
    run_name: str = "ttlab-run"


class PathsConfig(BaseModel):
    """Filesystem layout used by ttlab commands."""

    data_raw: Path
    data_processed: Path
    data_splits: Path
    out_metrics: Path
    out_artifacts: Path

    @field_validator("data_raw", "data_processed", "data_splits", "out_metrics", "out_artifacts", mode="before")
    @classmethod
    def _expand_path(cls, value: Any) -> Path:
        return Path(value).expanduser()

    def missing(self) -> List[Path]:
        """Return paths that do not exist yet."""

        return [
            path
            for path in [
                self.data_raw,
                self.data_processed,
                self.data_splits,
                self.out_metrics,
                self.out_artifacts,
            ]
            if not path.exists()
        ]


class DataConfig(BaseModel):
    """Dataset configuration."""

    format: DatasetFormat
    input_path: Path
    schema_fields: List[DatasetField]

    @field_validator("input_path", mode="before")
    @classmethod
    def _expand_input_path(cls, value: Any) -> Path:
        return Path(value).expanduser()

    def schema_dict(self) -> Dict[str, Any]:
        """Return the dataset schema as a serialisable dictionary."""

        return {
            "fields": [
                {
                    "name": field.name,
                    "type": field.type,
                    "required": field.required,
                }
                for field in self.schema_fields
            ]
        }

    def required_fields(self) -> Iterable[DatasetField]:
        """Iterate over required fields defined in the schema."""

        return (field for field in self.schema_fields if field.required)


class MLflowConfig(BaseModel):
    """Configuration for MLflow tracking."""

    tracking_uri: str
    experiment: str
    artifact_location: Optional[str] = None


class ProjectConfig(BaseModel):
    """Top-level project configuration."""

    general: GeneralConfig
    paths: PathsConfig
    data: DataConfig
    mlflow: MLflowConfig

    def missing_paths(self) -> List[Path]:
        """List configured directories that do not exist yet."""

        return self.paths.missing()


def load_config(path: Path | str) -> ProjectConfig:
    """Load a YAML configuration file and return a :class:`ProjectConfig`."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    return ProjectConfig.model_validate(data, strict=False)
