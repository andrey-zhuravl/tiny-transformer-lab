from __future__ import annotations

import pytest

pytest.importorskip("torch")

from ttlab.models.registry import create_model, list_models, register_model


def test_register_duplicate_raises() -> None:
    with pytest.raises(ValueError):
        
        @register_model("vanilla")
        class _Duplicate:  # pragma: no cover - definition only
            pass


def test_create_model_from_name() -> None:
    model = create_model("vanilla", vocab_size=64)
    assert model is not None
    assert "vanilla" in list_models()
