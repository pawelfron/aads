"""Shared constructor and forward contract for PubMed node classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class NodeClassifier(nn.Module, ABC):
    """Classifier over one homogeneous graph.

    Training builds every model with the arguments below and calls ``forward``
    on the full PubMed graph. Concrete architectures subclass this type and
    register a short name; this module does not implement an architecture.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Classify every node.

        ``x`` has shape ``[num_nodes, in_channels]`` and ``edge_index`` has
        shape ``[2, num_edges]``. The result has shape
        ``[num_nodes, out_channels]`` and stores class logits.
        """

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_layers={self.num_layers}, "
            f"dropout={self.dropout}"
        )
