"""
baseline_model.py

Baseline drug-pair interaction risk model, per proposal section 3.3.

Purpose: give you a working, evaluable result FAST (Day 2 of the plan),
and a measurable bar the GNN has to beat later.

Approach:
  1. Compute standard molecular descriptors for each drug (RDKit).
  2. For a drug PAIR, concatenate both drugs' descriptor vectors.
  3. Train a simple classifier (logistic regression or random forest)
     to predict interaction risk (binary: risky / not risky) from that
     concatenated vector.

This file is self-contained and runs on synthetic example data so you
can see the full pipeline work end-to-end today. Swap `build_training_table()`
for a real loader once you've pulled real pairs from DrugBank/SIDER.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# A small, fixed set of standard descriptors. Keep this list short and
# stable for the MVP -- more descriptors = more to debug, not necessarily
# more signal, when you have limited labelled pairs to train on.
DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "TPSA": Descriptors.TPSA,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
}


def drug_descriptors(smiles: str) -> np.ndarray:
    """Compute a fixed-length descriptor vector for one drug's SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: '{smiles}'")
    return np.array([func(mol) for func in DESCRIPTOR_FUNCS.values()], dtype=float)


def pair_features(smiles_a: str, smiles_b: str) -> np.ndarray:
    """
    Build the feature vector for a DRUG PAIR by concatenating both
    drugs' descriptor vectors. Order-independent: we sort so that
    (Drug A, Drug B) and (Drug B, Drug A) produce the same features.
    """
    vec_a = drug_descriptors(smiles_a)
    vec_b = drug_descriptors(smiles_b)
    # Sort the two vectors so pair order doesn't change the features.
    lo, hi = (vec_a, vec_b) if tuple(vec_a) <= tuple(vec_b) else (vec_b, vec_a)
    return np.concatenate([lo, hi])


@dataclass
class TrainedBaseline:
    model: object
    scaler: StandardScaler
    model_type: str

    def predict_risk(self, smiles_a: str, smiles_b: str) -> float:
        """Return the predicted probability that this pair is a risky interaction."""
        feats = pair_features(smiles_a, smiles_b).reshape(1, -1)
        feats_scaled = self.scaler.transform(feats)
        return float(self.model.predict_proba(feats_scaled)[0, 1])


def build_training_table(pairs: List[Tuple[str, str, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    pairs: list of (smiles_a, smiles_b, label) where label is 1 (known
    risky interaction) or 0 (no known interaction / safe pair).

    Replace this with a real loader over DrugBank/SIDER interaction
    pairs once you have them -- the model code below doesn't change.
    """
    X, y = [], []
    for smiles_a, smiles_b, label in pairs:
        X.append(pair_features(smiles_a, smiles_b))
        y.append(label)
    return np.array(X), np.array(y)


def train_baseline(pairs: List[Tuple[str, str, int]], model_type: str = "random_forest") -> Tuple[TrainedBaseline, dict]:
    """Train and evaluate the baseline model. Returns the trained model + metrics dict."""
    X, y = build_training_table(pairs)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }
    # AUC-ROC needs both classes present in the test set.
    if len(set(y_test)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_test, y_proba)
    else:
        metrics["auc_roc"] = None

    return TrainedBaseline(model=model, scaler=scaler, model_type=model_type), metrics


if __name__ == "__main__":
    # SYNTHETIC example pairs just to prove the pipeline works end to end.
    # Replace with real DrugBank DDI (drug-drug interaction) pairs + SIDER labels.
    drugs = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Warfarin": "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
        "Metformin": "CN(C)C(=N)N=C(N)N",
        "Simvastatin": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12",
    }

    # (drug_a, drug_b, label) -- 1 = known risky pair (illustrative, not medical fact),
    # 0 = assumed safe pair. THIS IS FAKE DATA for pipeline testing only.
    synthetic_pairs = [
        (drugs["Aspirin"], drugs["Warfarin"], 1),
        (drugs["Warfarin"], drugs["Ibuprofen"], 1),
        (drugs["Aspirin"], drugs["Ibuprofen"], 1),
        (drugs["Metformin"], drugs["Paracetamol"], 0),
        (drugs["Simvastatin"], drugs["Paracetamol"], 0),
        (drugs["Metformin"], drugs["Simvastatin"], 0),
        (drugs["Aspirin"], drugs["Paracetamol"], 0),
        (drugs["Ibuprofen"], drugs["Metformin"], 0),
    ] * 5  # repeated only so train_test_split has enough rows to run -- NOT how you'll train for real

    trained, metrics = train_baseline(synthetic_pairs, model_type="random_forest")
    print("Baseline metrics (on SYNTHETIC toy data -- replace with real DDI dataset):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    risk = trained.predict_risk(drugs["Aspirin"], drugs["Warfarin"])
    print(f"\nExample prediction — Aspirin + Warfarin risk score: {risk:.3f}")
