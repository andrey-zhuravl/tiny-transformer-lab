import pandas as pd
import pytest

from ttlab.config.models import DatasetField, DatasetFormat
from ttlab.data.validate import DatasetValidationError, validate_dataset


def _schema() -> list[DatasetField]:
    DF = DatasetField
    return [
        DF(field_name="id", type="str", required=True),
        DF(field_name="text", type="str", required=True),
        DF(field_name="task", type="str", required=True),
        DF(field_name="meta.grammar_rev", type="str", required=True),
        DF(field_name="meta.seed", type="int", required=True),
        DF(field_name="meta.split", type="str", required=True),
        DF(field_name="meta.template_index", type="int", required=True),
    ]

def test_validate_jsonl_success(tmp_path):
    df = pd.DataFrame(
        {"id":"train-lm-000000",
         "meta":{"grammar_rev":"c24837e4288880876a45ac6c452b0b659db4681300290dc331879ad2a7db2311",
                 "seed":"13",
                 "split":"train",
                 "template_index":"0"
                 },
         "task":"lm",
         "text":"Bob found a key at the market."
         }
    )
    path = tmp_path / "sample.jsonl"
    df.to_json(path, orient="records", lines=True)

    report = validate_dataset(path, _schema(), DatasetFormat.JSONL)
    assert report.rows == 4
    assert all(field.errors == [] for field in report.fields)


def test_validate_missing_required_column(tmp_path):
    df = pd.DataFrame({"id":"train-lm-000000",
         "meta":{"grammar_rev":"c24837e4288880876a45ac6c452b0b659db4681300290dc331879ad2a7db2311",
                 "seed":"13","split":"train","template_index":"0"
                 },
         "text":"Bob found a key at the market."
         })
    path = tmp_path / "bad.jsonl"
    df.to_json(path, orient="records", lines=True)

    with pytest.raises(DatasetValidationError):
        validate_dataset(path, _schema(), DatasetFormat.JSONL)

def test_validate_detects_nulls(tmp_path):
    df = pd.DataFrame(
        {"id": "train-lm-000000",
         "meta": {"grammar_rev": "c24837e4288880876a45ac6c452b0b659db4681300290dc331879ad2a7db2311",
                  "seed": "13", "split": "train", "template_index": "13"
                  },
         "text": "Bob found a key at the market."
         }
    )
    path = tmp_path / "nulls.jsonl"
    df.to_json(path, orient="records", lines=True)

    with pytest.raises(DatasetValidationError):
        validate_dataset(path, _schema(), DatasetFormat.JSONL)
