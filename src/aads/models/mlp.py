"""Feature-only MLP baseline for PubMed node classification."""

from __future__ import annotations

import torch
from torch import nn

from aads.models.base import NodeClassifier
from aads.models.registry import register_model


@register_model("mlp")
class MLP(NodeClassifier):
    """Classify each paper from its own features.

    ``num_layers`` is the number of linear maps. A one-layer model is a linear
    classifier. Deeper models use a ReLU and dropout after every layer except
    the last. The citation graph is ignored.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        widths = [in_channels, *([hidden_channels] * (num_layers - 1)), out_channels]
        self.linears = nn.ModuleList(
            nn.Linear(widths[index], widths[index + 1]) for index in range(num_layers)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        for linear in self.linears[:-1]:
            x = linear(x)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.linears[-1](x)
