"""Featurizes atoms in a molecule."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Optional, TypeVar, Union, cast
import warnings

import numpy as np  # type: ignore
import rdkit

from collagen.util import get_vdw_radius  # type: ignore

from ..types import AnyAtom

if TYPE_CHECKING:
    from collagen.core.molecules.mol import Mol
    from collagen.core.molecules.abstract_mol import AbstractAtom

# Type variable for any atom type
AnyAtom = TypeVar('AnyAtom')


class AtomFeaturizer(ABC):
    """Abstract AtomFeaturizer class. Other classes inherit this one. Invokes
    once per atom in a Mol.
    """

    @abstractmethod
    def __init__(self, layers: List[int], radii: Optional[List[float]] = None):
        """Initialize an AtomFeaturizer.

        Args:
            layers (List[int]): A list of atomic numbers. (?)
            radii (Optional[List[float]], optional): A list of radii. Defaults
                to None.
        """
        raise NotImplementedError()

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

    @abstractmethod
    def featurize(self, atom: AnyAtom) -> Tuple[int, float]:
        """Return a layer bitmask and a radius size for a given atom.

        Args:
            atom (AnyAtom): An atom.

        Returns:
            Tuple[int, float]: (bitmask, atom_radius)
        """
        raise NotImplementedError()

    @abstractmethod
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

    def featurize(
        self, atom: Union["rdkit.Chem.rdchem.Atom", "AbstractAtom"]
    ) -> Tuple[int, float]:
        """Feature an atom.

        Args:
            atom (rdkit.Chem.rdchem.Atom): An atom.

        Returns:
            Tuple[int, float]: (bitmask, atom_radius)
        """
        num: Union[int, None] = None

        if type(atom) is rdkit.Chem.rdchem.Atom:
            num = cast(int, atom.GetAtomicNum())  # type: ignore
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

class DeepFragAtomicNumFeaturizer(AtomFeaturizer):
    """A featurizer that creates voxel grids for specific atom types in receptors and ligands.
    
    For receptors: Creates separate grids for C, O, N, S and other heavy atoms (ignores H)
    For ligands: Creates separate grids for C, O, N and other heavy atoms (ignores H)
    """

    def __init__(self, layers: List[int], radii: Optional[List[float]] = None):
        """Initialize the featurizer with atom types and radii.

        Args:
            layers (List[int]): Must contain the atomic numbers in order:
                [6, 8, 7, 16] (C, O, N, S) for proper channel assignment
            radii (Optional[List[float]]): Optional list of radii for the atoms.
                If not provided, van der Waals radii will be used.
        """
        # Validate we have the expected atom types in expected order
        assert len(layers) == 4, "Must provide exactly 4 atomic numbers: C, O, N, S"
        assert layers[0] == 6, "First atomic number must be Carbon (6)"
        assert layers[1] == 7, "Second atomic number must be Nitrogen (7)" 
        assert layers[2] == 8, "Third atomic number must be Oxygen (8)"
        assert layers[3] == 16, "Fourth atomic number must be Sulfur (16)"

        self.layers = layers
        
        if radii is not None:
            assert len(radii) == 4, "Must provide exactly 4 radii values"
            self.radii = radii
        else:
            # Use van der Waals radii if none provided
            self.radii = [get_vdw_radius(num) for num in layers]

        # For receptor: C, O, N, S, other (no H) = 5 channels
        # For ligand: C, O, N, other (no H) = 4 channels
        # Total number of channels = 9
        self._num_features = 9

    def size(self) -> int:
        """Return the number of features produced by this featurizer.
        
        Returns:
            int: The number of features (9 total - 5 receptor + 4 ligand channels)
        """
        return self._num_features

    def featurize(self, atom: Union["rdkit.Chem.rdchem.Atom", "AbstractAtom"]) -> Tuple[int, float]:
        """Feature an atom.

        Args:
            atom (Union[rdkit.Chem.rdchem.Atom, AbstractAtom]): The atom to featurize

        Returns:
            Tuple[int, float]: (bitmask, atom_radius)
        """
        # Get atomic number
        if isinstance(atom, rdkit.Chem.rdchem.Atom):
            atomic_num = atom.GetAtomicNum()
        elif isinstance(atom, AbstractAtom):
            atomic_num = atom.num
        else:
            raise ValueError(f"Unknown atom type: {type(atom)}")

        # Skip hydrogen atoms
        if atomic_num == 1:
            return (0, 0)

        # Get radius - use van der Waals if not one of our specific atoms
        radius = self.radii[self.layers.index(atomic_num)] if atomic_num in self.layers else get_vdw_radius(atomic_num)
        
        # Check if this is a receptor atom based on the molecule metadata
        # This is handled in _get_mask()
        is_receptor = hasattr(atom, "GetOwningMol") and "Receptor" in getattr(atom.GetOwningMol(), "meta", {}).get("name", "")
        
        # Initialize mask to 0
        mask = self._get_mask(atomic_num, is_receptor)
            
        return (mask, radius)

    def _get_mask(self, atomic_num: int, is_receptor: bool) -> int:
        """Helper method to get the appropriate channel mask for an atom.
        
        Args:
            atomic_num (int): The atomic number
            is_receptor (bool): Whether this is a receptor atom
            
        Returns:
            int: The channel mask for this atom
        """
        if is_receptor:
            # Receptor channels are 0-4
            if atomic_num == 6:  # Carbon
                return 1 << 0
            elif atomic_num == 8:  # Oxygen
                return 1 << 1
            elif atomic_num == 7:  # Nitrogen
                return 1 << 2
            elif atomic_num == 16:  # Sulfur
                return 1 << 3
            else:
                # Other heavy atoms
                return 1 << 4
        else:
            # Ligand channels are 5-8
            if atomic_num == 6:  # Carbon
                return 1 << 5
            elif atomic_num == 8:  # Oxygen 
                return 1 << 6
            elif atomic_num == 7:  # Nitrogen
                return 1 << 7
            else:
                # Other heavy atoms
                return 1 << 8