"""Atom type definition."""

from typing import TYPE_CHECKING, Type

# https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING
# if TYPE_CHECKING:
import rdkit  # type: ignore
#    import collagen.core.molecules.abstract_mol

AnyAtom = Type[rdkit.Chem.rdchem.Atom]

# Union[
#     "rdkit.Chem.rdchem.Atom", "collagen.core.molecules.abstract_mol.AbstractAtom"
# ]
