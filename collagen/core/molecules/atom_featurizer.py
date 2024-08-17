"""Featurizes atoms in a molecule."""

from typing import TYPE_CHECKING, List, Any, Tuple, Optional, Union, cast
import warnings

import numpy as np
import rdkit

from ..types import AnyAtom

if TYPE_CHECKING:
    from collagen.core.molecules.mol import Mol
    from collagen.core.molecules.abstract_mol import AbstractAtom

class AtomFeaturizer(object):

    """Abstract AtomFeaturizer class. Other classes inherit this one. Invokes
    once per atom in a Mol.
    """

    def featurize_mol(self, mol: "Mol") -> Tuple["np.ndarray", "np.ndarray"]:
        """Featurize a Mol, returns (atom_mask, atom_radii).
        
        Args:
            mol (Mol): A molecule.
            
        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: (atom_mask, atom_radii)
        """
        ft = [self.featurize(a) for a in mol.atoms]
        atom_mask = np.array([x[0] for x in ft], dtype=np.int32)
        atom_radii = np.array([x[1] for x in ft], dtype=np.float32)
        return (atom_mask, atom_radii)

    def featurize(self, atom: AnyAtom) -> Tuple[int, float]:
        """Return a layer bitmask and a radius size for a given atom.

        Args:
            atom (AnyAtom): An atom.

        Returns:
            Tuple[int, float]: (bitmask, atom_radius)
        """
        raise NotImplementedError()

    def size(self) -> int:
        """Return the total number of layers.
        
        Returns:
            int: The total number of layers.
        """
        raise NotImplementedError()


class AtomicNumFeaturizer(AtomFeaturizer):

    """A featurizer that assigns each atomic number to a layer. By default,
    atomic radii are set to 1 however you can provide a radii argument of the
    same length as layers to assign separate atomic radii.
    """

    def __init__(self, layers: List[int], radii: Optional[List[float]] = None):
        """Initialize an AtomicNumFeaturizer.
        
        Args:
            layers (List[int]): A list of atomic numbers. (?)
            radii (Optional[List[float]], optional): A list of radii. Defaults
                to None.
        """
        assert len(layers) <= 32, "AtomicNumFeaturizer supports a maximum of 32 layers"
        self.layers = layers

        if radii is not None:
            assert len(layers) == len(
                radii
            ), "Must provide an equal number of radii as layers"
            self.radii = radii
        else:
            # Radii not provided, so assign radius 1 to all atoms.
            self.radii = [1] * len(self.layers)

    def featurize(self, atom: Union["rdkit.Chem.rdchem.Atom", "AbstractAtom"]) -> Tuple[int, float]:
        """Feature an atom.

        Args:
            atom (rdkit.Chem.rdchem.Atom): An atom.

        Returns:
            Tuple[int, float]: (bitmask, atom_radius)
        """
        num: Union[int, None] = None

        if type(atom) is rdkit.Chem.rdchem.Atom:
            num = cast(int, atom.GetAtomicNum()) # type: ignore
        elif type(atom).__name__ == "AbstractAtom":
            num = cast(int, atom.num)
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
        """Return the total number of layers.

        Returns:
            int: The total number of layers.
        """
        return len(self.layers)
