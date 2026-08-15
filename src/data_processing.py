"""
data_processing.py

Converts drug SMILES strings into molecular graph representations
that a Graph Neural Network can consume.

Core idea:
  - Atoms  -> graph nodes (with a feature vector describing the atom)
  - Bonds  -> graph edges (connecting two atom-nodes)

We use RDKit to parse the SMILES and extract atom/bond info, then
package it in a simple format that's easy to later convert into
PyTorch Geometric's `Data` object.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem


# A small, fixed vocabulary of atom types we care about.
# Anything outside this list is bucketed into "other" — keeps the
# feature vector small and consistent across all molecules.
ATOM_TYPES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "other"]


@dataclass
class MolecularGraph:
    """Simple container for a molecule's graph representation."""
    smiles: str
    atom_features: List[List[float]] = field(default_factory=list)   # one feature vector per atom
    edge_index: List[Tuple[int, int]] = field(default_factory=list)  # (source_atom, target_atom) pairs
    num_atoms: int = 0

    def summary(self) -> str:
        return (
            f"SMILES: {self.smiles}\n"
            f"  Atoms: {self.num_atoms}\n"
            f"  Bonds (directed edges): {len(self.edge_index)}\n"
            f"  Feature vector length per atom: {len(self.atom_features[0]) if self.atom_features else 0}"
        )


def _atom_to_features(atom: Chem.Atom) -> List[float]:
    """
    Turn one RDKit atom into a fixed-length numeric feature vector.

    This is deliberately simple for the MVP — you can enrich this later
    (hybridization, aromaticity, formal charge, chirality, etc.) once
    the baseline pipeline is working end to end.
    """
    symbol = atom.GetSymbol()
    one_hot = [1.0 if symbol == t else 0.0 for t in ATOM_TYPES[:-1]]
    is_other = 1.0 if symbol not in ATOM_TYPES[:-1] else 0.0
    one_hot.append(is_other)

    extra = [
        atom.GetDegree(),                       # number of directly-bonded neighbors
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),                    # attached hydrogens
        1.0 if atom.GetIsAromatic() else 0.0,
    ]
    return one_hot + extra


def smiles_to_graph(smiles: str) -> MolecularGraph:
    """
    Parse a SMILES string into a MolecularGraph.

    Raises ValueError if RDKit can't parse the SMILES (bad/malformed input).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: '{smiles}'")

    # Add explicit hydrogens off by default (keeps graphs smaller for MVP).
    atom_features = [_atom_to_features(atom) for atom in mol.GetAtoms()]

    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Add both directions since GNN message passing is typically undirected.
        edge_index.append((i, j))
        edge_index.append((j, i))

    return MolecularGraph(
        smiles=smiles,
        atom_features=atom_features,
        edge_index=edge_index,
        num_atoms=mol.GetNumAtoms(),
    )


if __name__ == "__main__":
    # Quick manual test with two real, well-known drugs.
    test_drugs = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Warfarin": "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O",
    }

    for name, smi in test_drugs.items():
        graph = smiles_to_graph(smi)
        print(f"--- {name} ---")
        print(graph.summary())
        print()
