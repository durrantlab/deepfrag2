"""Classes for abstract molecules."""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .mol import Mol
from ..types import AnyAtom


@dataclass
class AbstractAtom(object):

    """An abstract atom."""

    coord: "np.ndarray"
    num: Any = None
    metadata: Any = None


@dataclass
class AbstractBond(object):

    """An abstract bond."""

    edge: Tuple[int, int]
    metadata: Any = None


class AbstractMol(Mol):

    """An abstract molecule."""

    # TODO: Is this used anywhere?
    _atoms: List[AbstractAtom]
    _bonds: List[AbstractBond]

    def __init__(self, meta: Optional[dict] = None):
        """Initialize an abstract molecule.
        
        Args:
            meta (dict, optional): metadata for the molecule. Defaults to None.
        """
        if meta is None:
            meta = {}

        super(AbstractMol, self).__init__(meta=meta)
        self._atoms = []
        self._bonds = []

    def add_atom(self, atom: AbstractAtom) -> int:
        """
        Add an atom to the mol, returning the new atom index.

        Args:
            atom (AbstractAtom): The atom to add.

        Returns:
            int: The index of the newly added atom.
        """
        assert np.array(atom.coord).shape == (3,), "Atom coord must have shape (3,)"
        self._atoms.append(atom)
        return len(self._atoms) - 1

    def add_bond(self, bond: AbstractBond):
        """Add a bond to the mol.

        Args:
            bond (AbstractBond): The bond to add.
        """
        a, b = bond.edge
        assert a >= 0 and a < len(self._atoms), f"Atom index {a} is out of bounds"
        assert b >= 0 and b < len(self._atoms), f"Atom index {b} is out of bounds"
        self._bonds.append(bond)

    def sdf(self) -> str:
        """Generate a fake carbon skeleton SDF for visualization of abstract
        molecular topologies.
        
        Returns:
            str: The SDF string.
        """
        sdf = "\n\n\n"
        sdf += f"{len(self.coords):3d}{len(self._bonds):3d}  0  0  0  0  0  0  0  0999 V2000\n"

        for i in range(len(self.coords)):
            x, y, z = self.coords[i]
            sdf += (
                f"{x:10.4f}{y:10.4f}{z:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            )

        for bond in self._bonds:
            a, b = bond.edge
            sdf += f"{a+1:3d}{b+1:3d}  1  0  0  0  0\n"

        sdf += "M  END\n$$$$\n"
        return sdf

    @property
    def atoms(self) -> List[AnyAtom]:
        """Return the atoms in the molecule.
        
        Returns:
            List[AnyAtom]: The atoms in the molecule.
        """
        return self._atoms

    @property
    def coords(self) -> "np.ndarray":
        """Return the coordinates of the atoms in the molecule.
        
        Returns:
            numpy.ndarray: The coordinates of the atoms in the molecule.
        """
