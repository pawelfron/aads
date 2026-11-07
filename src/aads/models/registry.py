"""Name-based registry used by the training script to build a model."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from aads.models.base import NodeClassifier

ModelFactory = Callable[..., NodeClassifier]

REQUIRED_ARGUMENTS = (
    "in_channels",
    "hidden_channels",
    "out_channels",
    "num_layers",
    "dropout",
)

_MODELS: dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register a model class or factory under ``name``."""

    def decorator(factory: ModelFactory) -> ModelFactory:
        key = name.strip()
        if not key:
            raise ValueError("Model name must not be empty.")
        if key in _MODELS:
            raise ValueError(f"Model {key!r} is already registered.")
        _require_training_arguments(factory)
        _MODELS[key] = factory
        return factory

    return decorator


def available_models() -> list[str]:
    """Return registered model names in alphabetical order."""
    return sorted(_MODELS)


def build_model(
    name: str,
    *,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    dropout: float,
    **model_kwargs: object,
) -> NodeClassifier:
    """Construct a registered model with the shared training hyperparameters."""
    try:
        factory = _MODELS[name]
    except KeyError:
        known = ", ".join(available_models()) or "none"
        raise ValueError(f"Unknown model {name!r}. Registered models: {known}.") from None
    model = factory(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
        **model_kwargs,
    )
    if not isinstance(model, NodeClassifier):
        raise TypeError(
            f"Model {name!r} must return a NodeClassifier, got {type(model).__name__}."
        )
    return model


def _require_training_arguments(factory: ModelFactory) -> None:
    signature = inspect.signature(factory)
    parameters = list(signature.parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return
    missing: list[str] = []
    for name in REQUIRED_ARGUMENTS:
        parameter = signature.parameters.get(name)
        if parameter is None:
            missing.append(name)
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{_factory_name(factory)} must accept {name!r} as a keyword argument."
            )
    if missing:
        joined = ", ".join(missing)
        raise TypeError(
            f"{_factory_name(factory)} is missing constructor arguments: {joined}."
        )


def _factory_name(factory: ModelFactory) -> str:
    return getattr(factory, "__name__", type(factory).__name__)
