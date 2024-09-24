"""Atom type definition."""

from typing import TYPE_CHECKING, Type, Union
import rdkit

# if TYPE_CHECKING:
#    import rdkit  # type: ignore
#    import collagen.core.molecules.abstract_mol

AnyAtom = Type[rdkit.Chem.rdchem.Atom]

# Union[
#     "rdkit.Chem.rdchem.Atom", "collagen.core.molecules.abstract_mol.AbstractAtom"
# ]
