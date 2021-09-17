from typing import List

import numpy as np


class AtomFeaturizer(object):
    """Abstract AtomFeaturizer class. Invokes once per atom in a Mol."""

    def featurize_mol(self, mol: "Mol") -> "np.ndarray":
        """Featurize a Mol, returns a numpy array."""
        return np.array([self.featurize(a) for a in mol.atoms], dtype=np.int32)

    def featurize(self, atom: "rdkit.Chem.rdchem.Atom") -> int:
        """Returns a 32-bit layer bitmask for an RDKit atom."""
        raise NotImplementedError()

    def size(self) -> int:
        """Returns the total number of layers."""
        raise NotImplementedError()


class AtomicNumFeaturizer(AtomFeaturizer):
    """
    A featurizer that assigns each atomic number to a layer.

    Args:
        layers (List[int]): A list of atomic numbers.
    """

    def __init__(self, layers: List[int]):
        assert len(layers) <= 32
        self.layers = layers

    def featurize(self, atom: "rdkit.Chem.rdchem.Atom") -> int:
        num = atom.GetAtomicNum()
        if num in self.layers:
            return 1 << (self.layers.index(num))
        else:
            return 0

    def size(self) -> int:
        return len(self.layers)
