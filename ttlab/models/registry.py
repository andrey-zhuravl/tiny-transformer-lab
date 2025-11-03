"""Model registry utilities used across the training CLI."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, MutableMapping, Sequence, Union

from omegaconf import DictConfig, OmegaConf

from .base import BaseTTLabModel


_REGISTRY: Dict[str, Callable[..., BaseTTLabModel]] = {}


def register_model(name: str) -> Callable[[Callable[..., BaseTTLabModel]], Callable[..., BaseTTLabModel]]:
    """Register a model class or factory under the provided name."""

    def decorator(cls: Callable[..., BaseTTLabModel]) -> Callable[..., BaseTTLabModel]:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def list_models() -> Sequence[str]:
    """Return the registered model names in sorted order."""

    return list(sorted(_REGISTRY))


def _normalize_cfg(name_or_cfg: Union[str, Mapping[str, object], DictConfig]) -> Dict[str, object]:
    if isinstance(name_or_cfg, str):
        return {"model": {"kind": name_or_cfg}}
    if isinstance(name_or_cfg, DictConfig):
        container = OmegaConf.to_container(name_or_cfg, resolve=True)
        if not isinstance(container, MutableMapping):  # pragma: no cover - defensive
            raise TypeError("Resolved DictConfig must be a mapping to instantiate a model")
        return dict(container)
    if isinstance(name_or_cfg, Mapping):
        return dict(name_or_cfg)
    raise TypeError(
        "Expected model name as string or configuration mapping when creating a model"
    )


def create_model(
    name_or_cfg: Union[str, Mapping[str, object], DictConfig],
    *,
    vocab_size: int,
    **kwargs,
) -> BaseTTLabModel:
    """Instantiate the requested model using the registry."""

    cfg = _normalize_cfg(name_or_cfg)
    model_section = cfg.get("model", {})
    if not isinstance(model_section, Mapping):
        raise TypeError("Config 'model' section must be a mapping")
    kind = model_section.get("kind")
    if not isinstance(kind, str):
        raise KeyError("Model configuration must contain a string 'kind'")
    try:
        factory = _REGISTRY[kind]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown model kind: {kind!r}. Registered: {sorted(_REGISTRY)}") from exc
    return factory.from_config(cfg, vocab_size=vocab_size, **kwargs)
