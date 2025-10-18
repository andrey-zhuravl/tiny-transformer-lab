import pandas as pd
import pytest

from ttlab.config import DatasetField, DatasetFormat
from ttlab.data.validate import DatasetValidationError, validate_dataset


def _schema() -> list[DatasetField]:
    return [
        DatasetField(field_name="id", type="int", required=True),
        DatasetField(field_name="text", type="str", required=True),
        DatasetField(field_name="label", type="str", required=True),
    ]


def test_validate_jsonl_success(tmp_path):
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": ["hello", "world", "ttlab"],
            "label": ["a", "b", "c"],
        }
    )
    path = tmp_path / "sample.jsonl"
    df.to_json(path, orient="records", lines=True)

    report = validate_dataset(path, _schema(), DatasetFormat.JSONL)
    assert report.rows == 3
    assert all(field.errors == [] for field in report.fields)


def test_validate_missing_required_column(tmp_path):
    df = pd.DataFrame({"id": [1], "text": ["oops"]})
    path = tmp_path / "bad.jsonl"
    df.to_json(path, orient="records", lines=True)

    with pytest.raises(DatasetValidationError):
        validate_dataset(path, _schema(), DatasetFormat.JSONL)

def test_validate_detects_nulls(tmp_path):
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "text": ["ok", None],
            "label": ["x", "y"],
        }
    )
    path = tmp_path / "nulls.jsonl"
    df.to_json(path, orient="records", lines=True)

    with pytest.raises(DatasetValidationError):
        validate_dataset(path, _schema(), DatasetFormat.JSONL)
