"""Atom type definition."""

from typing import Union
AnyAtom = Union["rdkit.Chem.rdchem.Atom", "collagen.core.molecules.abstract_mol.AbstractAtom"]
