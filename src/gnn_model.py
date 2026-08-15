"""
gnn_model.py

A Graph Neural Network that learns a fixed-length "embedding" vector
for a drug, directly from its molecular graph (atoms + bonds) --
no hand-picked descriptors needed, unlike the baseline model.

Architecture (standard, well-established pattern for molecule GNNs):
  1. Several GCN (Graph Convolutional Network) layers -- each lets
     every atom "absorb" information from its directly bonded
     neighbors. Stacking layers lets information flow further
     (2 layers = each atom sees its neighbors' neighbors, etc.)
  2. A pooling step that collapses all the atom-level vectors for
     one molecule into a SINGLE fixed-length vector (the molecule's
     "embedding") -- since molecules have different numbers of
     atoms, but the model output needs to be a fixed size.
  3. For a drug PAIR: get both drugs' embeddings, concatenate them,
     and feed through a small classifier head to predict interaction
     risk (this is the "fusion" step from Day 5 of the plan).

This file also includes the PyTorch Geometric Data conversion, since
MolecularGraph (from data_processing.py) is a plain Python object and
PyG needs its own Data/Batch format for training.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from data_processing import MolecularGraph, smiles_to_graph


def molgraph_to_pyg_data(graph: MolecularGraph) -> Data:
    """Convert our MolecularGraph into a PyTorch Geometric Data object."""
    x = torch.tensor(graph.atom_features, dtype=torch.float)

    if len(graph.edge_index) == 0:
        # Single-atom molecules (rare, but guard against it): no edges.
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        # PyG expects edge_index shaped [2, num_edges], not a list of pairs.
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


class DrugGNNEncoder(nn.Module):
    """
    Encodes a single molecular graph into a fixed-length embedding vector.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, embedding_dim)

    def forward(self, x, edge_index, batch):
        # Each GCN layer: every atom aggregates info from its bonded neighbors.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # Collapse per-atom vectors into ONE vector per molecule (mean pooling).
        # `batch` tells PyG which atoms belong to which molecule when we
        # process several molecules together (a "batch") for efficiency.
        molecule_embedding = global_mean_pool(x, batch)
        return molecule_embedding


class DrugPairInteractionModel(nn.Module):
    """
    Full model: encodes BOTH drugs in a pair, fuses their embeddings,
    and predicts interaction risk (a single probability).
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, embedding_dim: int = 64):
        super().__init__()
        # Both drugs share the SAME encoder -- the model shouldn't
        # care which drug is "first" vs "second."
        self.encoder = DrugGNNEncoder(in_channels, hidden_channels, embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, batch_a: Batch, batch_b: Batch) -> torch.Tensor:
        emb_a = self.encoder(batch_a.x, batch_a.edge_index, batch_a.batch)
        emb_b = self.encoder(batch_b.x, batch_b.edge_index, batch_b.batch)

        fused = torch.cat([emb_a, emb_b], dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits  # raw logits -- apply sigmoid for a 0-1 probability


if __name__ == "__main__":
    # Quick sanity check: run two real drugs through the model and confirm
    # shapes come out right. Weights are untrained/random at this point --
    # this just proves the architecture is wired up correctly.
    aspirin_graph = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")
    warfarin_graph = smiles_to_graph("CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O")

    data_a = molgraph_to_pyg_data(aspirin_graph)
    data_b = molgraph_to_pyg_data(warfarin_graph)

    # Wrap single graphs into a "batch of 1" -- required by PyG's pooling ops.
    batch_a = Batch.from_data_list([data_a])
    batch_b = Batch.from_data_list([data_b])

    in_channels = data_a.x.shape[1]  # feature vector length per atom
    model = DrugPairInteractionModel(in_channels=in_channels)

    logits = model(batch_a, batch_b)
    probability = torch.sigmoid(logits)

    print(f"Input feature size per atom: {in_channels}")
    print(f"Raw logit: {logits.item():.4f}")
    print(f"Predicted interaction probability (untrained, random weights): {probability.item():.4f}")
    print("\nModel architecture wired up correctly." if logits.shape == (1,) else "Shape mismatch!")
