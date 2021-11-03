from typing import List, Any, Tuple, Optional
import warnings

import numpy as np
import rdkit

from .types import AnyAtom


class AtomFeaturizer(object):
    """Abstract AtomFeaturizer class. Invokes once per atom in a Mol."""

    def featurize_mol(self, mol: "Mol") -> Tuple["numpy.ndarray", "numpy.ndarray"]:
        """Featurize a Mol, returns (atom_mask, atom_radii)."""
        ft = [self.featurize(a) for a in mol.atoms]
        atom_mask = np.array([x[0] for x in ft], dtype=np.int32)
        atom_radii = np.array([x[1] for x in ft], dtype=np.float32)
        return (atom_mask, atom_radii)

    def featurize(self, atom: AnyAtom) -> Tuple[int, float]:
        """
        Returns a layer bitmask and a radius size for a given atom.

        Args:
            atom (AnyAtom): An atom.

        Returns:
            Tuple[int, float]: (bitmask, atom_radius)
        """
        raise NotImplementedError()

    def size(self) -> int:
        """Returns the total number of layers."""
        raise NotImplementedError()


class AtomicNumFeaturizer(AtomFeaturizer):
    """
    A featurizer that assigns each atomic number to a layer. By default, atomic radii are set to 1 however
    you can provide a radii argument of the same length as layers to assign separate atomic radii.

    Args:
        layers (List[int]): A list of atomic numbers.
        radii (List[float], optional): An optional list of atomic radii to use (of the same length as layers).
    """

    def __init__(self, layers: List[int], radii: Optional[List[float]] = None):
        assert len(layers) <= 32, "AtomicNumFeaturizer supports a maximum of 32 layers"
        self.layers = layers

        if radii is not None:
            assert len(layers) == len(
                radii
            ), "Must provide an equal number of radii as layers"
            self.radii = radii
        else:
            self.radii = [1] * len(self.layers)

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
            idx = self.layers.index(num)
            return (1 << idx, self.radii[idx])
        else:
            return (0, 0)

    def size(self) -> int:
        return len(self.layers)
