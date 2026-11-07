"""Registered graph classifiers for PubMed.

Training builds models through :func:`build_model` and does not import an
architecture module itself. Define a :class:`NodeClassifier`, register it
with :func:`register_model` in the module that owns it, and import that
module below so registration runs on package import.

No classifiers are registered yet.
"""

from aads.models.base import NodeClassifier
from aads.models.registry import available_models, build_model, register_model

# Import concrete model modules below so their @register_model decorators run.

__all__ = [
    "NodeClassifier",
    "available_models",
    "build_model",
    "register_model",
]
