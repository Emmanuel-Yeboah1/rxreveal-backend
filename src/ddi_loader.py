"""
ddi_loader.py

Loads REAL drug-drug interaction data (DrugBank DDI, via the TDC
"Therapeutics Data Commons" library) and converts it into the
(smiles_a, smiles_b, label) format that baseline_model.py and the
GNN pipeline expect.

Why TDC: it bundles the DrugBank interaction pairs WITH each drug's
SMILES string already attached, and downloads automatically the first
time you run this -- no manual DrugBank account/license application
needed, which matters given your timeline.

First run will download the dataset (a few MB) into a local ./data
folder -- this can take a minute or two depending on your connection.
"""

import random
from typing import List, Tuple

from tdc.multi_pred import DDI


def load_drugbank_ddi_pairs(
    max_positive_pairs: int = 3000,
    negative_ratio: float = 1.0,
    random_seed: int = 42,
) -> List[Tuple[str, str, int]]:
    """
    Load real DrugBank DDI pairs and return them as a binary-labeled
    list: (smiles_a, smiles_b, label) where label=1 means "known
    interaction" and label=0 means "no known interaction" (a randomly
    sampled pair not present in the known-interaction list).

    max_positive_pairs: cap how many real positive pairs to use.
        The full dataset has ~191,808 pairs -- more than you need
        for an MVP and slower to train on. Start smaller, scale up
        if you have time.
    negative_ratio: how many negative pairs to generate per positive
        pair. 1.0 = balanced dataset (recommended for the MVP).
    """
    random.seed(random_seed)

    print("Loading DrugBank DDI dataset (downloads on first run)...")
    data = DDI(name="DrugBank")
    df = data.get_data()  # columns: Drug1_ID, Drug1, Drug2_ID, Drug2, Y

    # Cap the number of positive pairs used, for faster iteration.
    if len(df) > max_positive_pairs:
        df = df.sample(n=max_positive_pairs, random_state=random_seed)

    positive_pairs = [
        (row["Drug1"], row["Drug2"], 1) for _, row in df.iterrows()
    ]

    # Build a set of all known-interacting SMILES pairs (both orders)
    # so we don't accidentally sample a "negative" that's actually
    # a real interaction.
    known_pairs = set()
    for smiles_a, smiles_b, _ in positive_pairs:
        known_pairs.add((smiles_a, smiles_b))
        known_pairs.add((smiles_b, smiles_a))

    all_drug_smiles = list(set(df["Drug1"]).union(set(df["Drug2"])))

    num_negatives_needed = int(len(positive_pairs) * negative_ratio)
    negative_pairs = []
    attempts = 0
    max_attempts = num_negatives_needed * 20  # safety valve

    while len(negative_pairs) < num_negatives_needed and attempts < max_attempts:
        attempts += 1
        a, b = random.sample(all_drug_smiles, 2)
        if (a, b) not in known_pairs:
            negative_pairs.append((a, b, 0))
            known_pairs.add((a, b))  # avoid picking it twice
            known_pairs.add((b, a))

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    print(f"Loaded {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs.")
    return all_pairs


if __name__ == "__main__":
    pairs = load_drugbank_ddi_pairs(max_positive_pairs=500, negative_ratio=1.0)
    print(f"\nTotal pairs ready for training: {len(pairs)}")
    print("Example pair:", pairs[0])
