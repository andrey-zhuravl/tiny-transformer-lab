import pandas as pd

from ttlab.config.models import DatasetFormat
from ttlab.data.stats import compute_stats


def test_compute_stats_includes_null_fraction(tmp_path):
    df = pd.DataFrame({"id": [1, 2, 3], "text": ["a", "bb", None]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path, index=False)

    stats = compute_stats(path, DatasetFormat.PARQUET, text_fields=["text"], key_field="id")
    assert stats["rows"] == 3
    assert "null_fraction" in stats["columns_stats"]["text"]
    assert stats["unique_keys"] == 3
