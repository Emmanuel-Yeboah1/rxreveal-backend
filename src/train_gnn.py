"""
train_gnn.py

Trains the DrugPairInteractionModel (GNN) on real DrugBank DDI pairs,
and reports the same metrics as the baseline so you can directly
compare the two (Day 6 of the plan).
"""

import random

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch

from data_processing import smiles_to_graph
from ddi_loader import load_drugbank_ddi_pairs
from gnn_model import DrugPairInteractionModel, molgraph_to_pyg_data


def build_batches(pairs, batch_size=32):
    """
    Convert (smiles_a, smiles_b, label) pairs into batched PyG Batch
    objects, ready to feed the model. Skips any pair RDKit can't parse
    rather than crashing the whole run.
    """
    graph_cache = {}  # avoid re-parsing the same SMILES repeatedly

    def get_graph(smiles):
        if smiles not in graph_cache:
            try:
                graph_cache[smiles] = molgraph_to_pyg_data(smiles_to_graph(smiles))
            except ValueError:
                graph_cache[smiles] = None
        return graph_cache[smiles]

    data_a_list, data_b_list, labels = [], [], []
    for smiles_a, smiles_b, label in pairs:
        da, db = get_graph(smiles_a), get_graph(smiles_b)
        if da is None or db is None:
            continue  # skip unparseable SMILES
        data_a_list.append(da)
        data_b_list.append(db)
        labels.append(label)

    batches = []
    for i in range(0, len(data_a_list), batch_size):
        chunk_a = data_a_list[i : i + batch_size]
        chunk_b = data_b_list[i : i + batch_size]
        chunk_labels = labels[i : i + batch_size]
        batches.append((
            Batch.from_data_list(chunk_a),
            Batch.from_data_list(chunk_b),
            torch.tensor(chunk_labels, dtype=torch.float),
        ))
    return batches


def evaluate(model, batches):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch_a, batch_b, labels in batches:
            logits = model(batch_a, batch_b)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
    }
    if len(set(all_labels)) > 1:
        metrics["auc_roc"] = roc_auc_score(all_labels, all_probs)
    return metrics


def train():
    random.seed(42)
    torch.manual_seed(42)

    print("Loading real DrugBank DDI data...")
    pairs = load_drugbank_ddi_pairs(max_positive_pairs=4000, negative_ratio=1.0)

    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

    print("Converting SMILES to molecular graphs (this may take a minute)...")
    train_batches = build_batches(train_pairs, batch_size=32)
    test_batches = build_batches(test_pairs, batch_size=32)
    print(f"Train batches: {len(train_batches)}, Test batches: {len(test_batches)}")

    # Feature size comes from the first successfully parsed molecule.
    in_channels = train_batches[0][0].x.shape[1]
    model = DrugPairInteractionModel(in_channels=in_channels)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = 60
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_a, batch_b, labels in train_batches:
            optimizer.zero_grad()
            logits = model(batch_a, batch_b)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_batches)
        if epoch % 5 == 0 or epoch == 1:
            test_metrics = evaluate(model, test_batches)
            print(f"Epoch {epoch:2d} | train loss: {avg_loss:.4f} | "
                  f"test accuracy: {test_metrics['accuracy']:.3f} | "
                  f"test AUC-ROC: {test_metrics.get('auc_roc', float('nan')):.3f}")

    print("\nFinal GNN metrics on real DrugBank DDI data:")
    final_metrics = evaluate(model, test_batches)
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")

    torch.save(model.state_dict(), "gnn_model_weights.pt")
    print("\nModel weights saved to gnn_model_weights.pt")


if __name__ == "__main__":
    train()
