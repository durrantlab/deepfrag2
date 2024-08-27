"""Atom type definition."""

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import rdkit
    import collagen.core.molecules.abstract_mol

AnyAtom = Union[
    "rdkit.Chem.rdchem.Atom", "collagen.core.molecules.abstract_mol.AbstractAtom"
]
