"""Registered graph classifiers for PubMed.

Training builds models through :func:`build_model` and does not import an
architecture module itself. Define a :class:`NodeClassifier`, register it
with :func:`register_model` in the module that owns it, and import that
module below so registration runs on package import.
"""

from aads.models.base import NodeClassifier
from aads.models.mlp import MLP
from aads.models.registry import available_models, build_model, register_model

__all__ = [
    "MLP",
    "NodeClassifier",
    "available_models",
    "build_model",
    "register_model",
]
