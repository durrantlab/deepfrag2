from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdmolops

# From https://www.rdkit.org/docs/Cookbook.html
def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol

def standardize_smiles(smiles: str) -> str:
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
            canonical=True  # e.g., all benzenes are written as aromatic
        )

    except Exception as e:
        print(f"CAUGHT EXCEPTION: Could not standardize SMILES: {smiles} >> ", e)
        return smiles
