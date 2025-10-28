from ttlab.utils.mlflow_utils import _flatten, _parse_tags


def test_flatten_simple() -> None:
    payload = {"a": 1, "b": {"c": 2}}
    assert _flatten(payload) == {"a": 1, "b.c": 2}


def test_parse_tags() -> None:
    tags = _parse_tags(["k1=v1", "k2=v2"])
    assert tags == {"k1": "v1", "k2": "v2"}
