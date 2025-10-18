import pandas as pd

from ttlab.data.split import split_dataset


def test_split_by_row_deterministic(tmp_path):
    df = pd.DataFrame({"id": range(30), "text": ["a"] * 30})
    src = tmp_path / "data.jsonl"
    df.to_json(src, orient="records", lines=True)

    stats_first = split_dataset(src, tmp_path / "splits1", (0.6, 0.2, 0.2), seed=123)
    stats_second = split_dataset(src, tmp_path / "splits2", (0.6, 0.2, 0.2), seed=123)

    assert stats_first["train_rows"] == stats_second["train_rows"]
    assert stats_first["val_rows"] == stats_second["val_rows"]
    assert stats_first["test_rows"] == stats_second["test_rows"]

def test_split_by_key(tmp_path):
    df = pd.DataFrame({"id": [1, 1, 2, 2, 3, 3], "value": list(range(6))})
    src = tmp_path / "data.parquet"
    df.to_parquet(src, index=False)

    stats = split_dataset(src, tmp_path / "splits", (0.34, 0.33, 0.33), seed=42, strategy="by_key", key_column="id")
    assert stats["format"] == "PARQUET"
    assert (tmp_path / "splits" / "train.parquet").exists()
