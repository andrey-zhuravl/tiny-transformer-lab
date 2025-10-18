import json

import pandas as pd

from ttlab.config import DatasetFormat
from ttlab.data.convert import convert_dataset


def test_convert_jsonl_to_parquet(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "text": ["a", "b"]})
    src = tmp_path / "sample.jsonl"
    dst = tmp_path / "sample.parquet"
    df.to_json(src, orient="records", lines=True)

    result = convert_dataset(src, dst, DatasetFormat.PARQUET)
    assert dst.exists()
    assert result["rows"] == 2

    roundtrip_path = tmp_path / "roundtrip.jsonl"
    back = convert_dataset(dst, roundtrip_path, DatasetFormat.JSONL)
    with roundtrip_path.open() as fh:
        first_record = json.loads(fh.readline())
    assert first_record["id"] == 1
    assert back["format"] == DatasetFormat.JSONL.value
