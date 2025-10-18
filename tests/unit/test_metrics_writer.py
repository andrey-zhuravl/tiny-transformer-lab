import json

from ttlab.observability.metrics_writer import MetricsWriter


def test_metrics_writer_flush(tmp_path):
    path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(path, default_run_id="test-run")
    writer.log(event="convert", metric="rows", value=10)
    writer.flush()

    with path.open() as fh:
        record = json.loads(fh.readline())
    assert record["metric"] == "rows"
    assert record["run_id"] == "test-run"
