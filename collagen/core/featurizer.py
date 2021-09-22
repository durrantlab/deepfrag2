from typing import List, Any
import warnings

import numpy as np
import rdkit

from .types import AnyAtom


class AtomFeaturizer(object):
    """Abstract AtomFeaturizer class. Invokes once per atom in a Mol."""

    def featurize_mol(self, mol: "Mol") -> "np.ndarray":
        """Featurize a Mol, returns a numpy array."""
        return np.array([self.featurize(a) for a in mol.atoms], dtype=np.int32)

    def featurize(self, atom: AnyAtom) -> int:
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

    def featurize(self, atom: Any) -> int:
        num = None

        if type(atom) is rdkit.Chem.rdchem.Atom:
            num = atom.GetAtomicNum()
        elif type(atom).__name__ == "AbstractAtom":
            num = atom.num
        else:
            warnings.warn(
                f"Unknown unknown atom type {type(atom)} in AtomicNumFeaturizer"
            )

        if num is None:
            warnings.warn("Atom type is None in AtomicNumFeaturizer.featurize")

        if num in self.layers:
            return 1 << (self.layers.index(num))
        else:
            return 0

    def size(self) -> int:
        return len(self.layers)
