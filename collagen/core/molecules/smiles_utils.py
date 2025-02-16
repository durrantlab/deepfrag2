"""SMILES utilities."""

from typing import Union
from rdkit import Chem  # type: ignore
from rdkit.Chem import rdmolops  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from rdkit.Chem.rdmolops import RemoveHs  # type: ignore
from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore

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

        # Fix a few common problems. (Code taken from sanitize_neutralize_standardize_mol, grid_ml_base repo)
        replacements = [
            # Check if structure contains [P+](=O)=O. If so, replace that with P(=O)O.
            ("P(=O)=[O]", "P(=O)O"),
            
            # *S(O)(O)O is clearly sulfonate. Just poorly processed. S(O)(O)O ->
            # S(=O)(=O)O
            ("S([O;$([O-,OH])])([O;$([O-,OH])])([O;$([O-,OH])])", "S(=O)(=O)O"),

            # *P(O)(O)O is clearly phosphate. Just poorly processed. P(O)(O)O ->
            # P(=O)(O)O
            ("P([O;$([O-,OH])])([O;$([O-,OH])])([O;$([O-,OH])])", "P(=O)(O)O"),
            
            # Terminal geminal diols are going to be carboxylates.
            ("[C;H1]([O;$([O-,OH])])([O;$([O-,OH])])", "C(=O)(O)"),

            # Same with terminal geminol thiols
            ("[S;H1]([O;$([O-,OH])])([O;$([O-,OH])])", "S(=O)(O)"),
        ]

        for patt1, patt2 in replacements:
            ps = AllChem.ReplaceSubstructs(
                rdmol, Chem.MolFromSmarts(patt1), Chem.MolFromSmarts(patt2)
            )
            if ps:
                rdmol = ps[0]
                # Update properties after each replacement
                for atom in rdmol.GetAtoms():
                    atom.UpdatePropertyCache(strict=False)

        # Neutralize the molecule (charges)
        neutralize_atoms(rdmol)

        # rdmolops.Cleanup(rdmol)
        # rdmolops.RemoveStereochemistry(rdmol)

        # Remove hydrogens and update properties
        rdmol = RemoveHs(rdmol, sanitize=False)
        for atom in rdmol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)

        # Initial sanitization with subset of operations
        sanitize_flags = Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES
        Chem.SanitizeMol(rdmol, sanitizeOps=sanitize_flags)
        
        # Update properties before full sanitization
        for atom in rdmol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
            
        # Full sanitization
        Chem.SanitizeMol(rdmol)

        # Cleanup molecule
        rdmol = rdMolStandardize.Cleanup(rdmol)
        for atom in rdmol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)

        # Normalize molecule
        norm = rdMolStandardize.Normalizer()
        rdmol = norm.normalize(rdmol)
        for atom in rdmol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)

        # Uncharge molecule
        uncharger = rdMolStandardize.Uncharger()
        rdmol = uncharger.uncharge(rdmol)
        for atom in rdmol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)

        # Canonicalize tautomers
        enumerator = rdMolStandardize.TautomerEnumerator()
        rdmol = enumerator.Canonicalize(rdmol)
        for atom in rdmol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)

        # Clean up stereochemistry
        # Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        rdmolops.RemoveStereochemistry(rdmol)

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
