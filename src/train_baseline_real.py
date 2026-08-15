"""
train_baseline_real.py

Trains the baseline model on REAL DrugBank DDI data instead of the
synthetic toy pairs. Run this once ddi_loader.py is working.
"""

from ddi_loader import load_drugbank_ddi_pairs
from baseline_model import train_baseline


if __name__ == "__main__":
    # Start small so you can iterate fast; raise max_positive_pairs
    # later once everything works end-to-end.
    pairs = load_drugbank_ddi_pairs(max_positive_pairs=4000, negative_ratio=1.0)

    trained, metrics = train_baseline(pairs, model_type="random_forest")

    print("\nBaseline metrics on REAL DrugBank DDI data:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

