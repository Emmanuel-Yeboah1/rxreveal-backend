"""
case_study_check.py

Validates both the baseline and GNN models against a known, real-world
drug interaction: Warfarin + Aspirin, which are well-documented to
increase bleeding risk when taken together (per the proposal's
required case study).

This doesn't replace the aggregate metrics (accuracy/precision/recall/
AUC-ROC) -- it's a concrete, explainable sanity check you can point to
directly in your report: "does the model correctly flag a real,
known-dangerous pair?"
"""

import torch
from torch_geometric.data import Batch

from baseline_model import train_baseline
from data_processing import smiles_to_graph
from ddi_loader import load_drugbank_ddi_pairs
from gnn_model import DrugPairInteractionModel, molgraph_to_pyg_data

WARFARIN_SMILES = "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O"
ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"


def check_baseline():
    print("=== Baseline model ===")
    pairs = load_drugbank_ddi_pairs(max_positive_pairs=4000, negative_ratio=1.0)
    trained, metrics = train_baseline(pairs, model_type="random_forest")
    risk = trained.predict_risk(ASPIRIN_SMILES, WARFARIN_SMILES)
    print(f"Aspirin + Warfarin predicted risk: {risk:.3f}")
    print(f"Flagged as risky (>0.5)? {'YES' if risk > 0.5 else 'NO'}")
    return risk


def check_gnn(weights_path="gnn_model_weights.pt"):
    print("\n=== GNN model ===")
    aspirin_data = molgraph_to_pyg_data(smiles_to_graph(ASPIRIN_SMILES))
    warfarin_data = molgraph_to_pyg_data(smiles_to_graph(WARFARIN_SMILES))

    batch_a = Batch.from_data_list([aspirin_data])
    batch_b = Batch.from_data_list([warfarin_data])

    in_channels = aspirin_data.x.shape[1]
    model = DrugPairInteractionModel(in_channels=in_channels)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    with torch.no_grad():
        logits = model(batch_a, batch_b)
        risk = torch.sigmoid(logits).item()

    print(f"Aspirin + Warfarin predicted risk: {risk:.3f}")
    print(f"Flagged as risky (>0.5)? {'YES' if risk > 0.5 else 'NO'}")
    return risk


if __name__ == "__main__":
    baseline_risk = check_baseline()
    gnn_risk = check_gnn()

    print("\n=== Summary ===")
    print(f"Baseline risk score: {baseline_risk:.3f}")
    print(f"GNN risk score:      {gnn_risk:.3f}")
