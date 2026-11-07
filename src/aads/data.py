"""PubMed citation network used for scientific-field classification."""

from __future__ import annotations

from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

PUBMED_NAME = "PubMed"
SPLITS = ("public", "full", "random")

# Column order of the one-hot labels shipped with the Planetoid PubMed graph.
CLASS_NAMES: tuple[str, ...] = (
    "Diabetes Mellitus, Experimental",
    "Diabetes Mellitus Type 1",
    "Diabetes Mellitus Type 2",
)


def load_pubmed(
    root: str | Path,
    *,
    split: str = "public",
    num_train_per_class: int = 20,
    num_val: int = 500,
    num_test: int = 1000,
) -> tuple[Planetoid, Data]:
    """Load the Planetoid PubMed graph.

    Each node is a paper, edges are citations, and the node label is the
    paper's scientific field. ``split="public"`` is the fixed split from the
    Planetoid benchmark. ``split="random"`` draws a split with the given
    sizes from the current torch random generator, so seed the generator
    before calling this function.
    """
    if split not in SPLITS:
        choices = ", ".join(SPLITS)
        raise ValueError(f"Unsupported split {split!r}. Expected one of: {choices}.")
    dataset = Planetoid(
        root=str(root),
        name=PUBMED_NAME,
        split=split,
        num_train_per_class=num_train_per_class,
        num_val=num_val,
        num_test=num_test,
    )
    return dataset, dataset[0]


def split_sizes(data: Data) -> dict[str, int]:
    """Return the number of nodes in the train, validation, and test masks."""
    if data.num_nodes is None:
        raise ValueError("The graph does not know how many nodes it has.")
    sizes: dict[str, int] = {}
    for split in ("train", "val", "test"):
        mask = data[f"{split}_mask"]
        if mask.dim() != 1 or mask.shape[0] != data.num_nodes:
            raise ValueError(
                f"{split}_mask has shape {tuple(mask.shape)}; expected [{data.num_nodes}]."
            )
        sizes[split] = int(mask.sum().item())
    return sizes
