"""SMILES utilities."""

from typing import Union
from rdkit import Chem  # type: ignore
from rdkit.Chem import rdmolops  # type: ignore


# From https://www.rdkit.org/docs/Cookbook.html
def neutralize_atoms(mol: Chem.Mol) -> Chem.Mol:
    """Neutralize the molecule by adding/removing hydrogens.

    Args:
        mol (Chem.Mol): RDKit molecule.

    Returns:
        Chem.Mol: Neutralized RDKit molecule.
    """
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if at_matches_list:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def standardize_smiles(smiles: str, none_if_fails: bool = False) -> Union[str, None]:
    """Standardize SMILES string.

    Args:
        smiles (str): SMILES string.
        none_if_fails (bool): If True, will return None.

    Returns:
        str: Standardized SMILES string. Returns None if `none_if_fails` is
            True and the standardization fails.
    """
    # Catch all errors
    try:
        # Convert smiles to rdkit mol
        rdmol = Chem.MolFromSmiles(smiles)

        # Neutralize the molecule (charges)
        neutralize_atoms(rdmol)

        rdmolops.Cleanup(rdmol)
        rdmolops.RemoveStereochemistry(rdmol)
        rdmolops.SanitizeMol(rdmol, catchErrors=True)

        # Remove hydrogens
        rdmol = Chem.RemoveHs(rdmol)

        return Chem.MolToSmiles(
            rdmol,
            isomericSmiles=False,  # No chirality
            canonical=True,  # e.g., all benzenes are written as aromatic
        )

    except Exception as e:
        if none_if_fails:
            return None
        else:
            print(f"CAUGHT EXCEPTION: Could not standardize SMILES: {smiles} >> ", e)
            return smiles
