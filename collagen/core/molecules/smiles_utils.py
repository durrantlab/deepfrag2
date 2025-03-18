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


def standardize_smiles_or_rdmol(smiles_or_mol: Union[str, Chem.Mol], none_if_fails: bool = False) -> Union[Union[str, Chem.Mol], None]:
    """Standardize SMILES string.

    Args:
        smiles_or_mol (str): SMILES string.
        none_if_fails (bool): If True, will return None.

    Returns:
        Union[str, Chem.Mol]: Standardized SMILES string, if input was SMILES.
            Otherwise, stanardized Chem.Mol. Returns None if `none_if_fails` is
            True and the standardization fails.
    """
    # Convert smiles to rdkit mol
    if isinstance(smiles_or_mol, str):
        rdmol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        # It's the mol, but make a copy
        rdmol = Chem.Mol(smiles_or_mol)

    # Catch all errors
    try:

        # Fix a few common problems. (Code taken from sanitize_neutralize_standardize_mol, grid_ml_base repo)
        replacements = [
            # Check if structure contains [P+](=O)=O. If so, replace that with P(=O)O.
            ("[P+1:1](=O)=[O]", "[P+0:1](=O)O"),  # Example: "CCCC[P+](=O)=O"
            
            # *S(O)(O)O is clearly sulfonate. Just poorly processed. S(O)(O)O ->
            # S(=O)(=O)O
            ("[S:1]([O;$([O-,OH])])([O;$([O-,OH])])([O;$([O-,OH])])", "[S:1](=O)(=O)O"),

            # *P(O)(O)O is clearly phosphate. Just poorly processed. P(O)(O)O ->
            # P(=O)(O)O
            ("[P:1]([O;$([O-,OH])])([O;$([O-,OH])])([O;$([O-,OH])])", "[P:1](=O)(O)O"),

            # [PH](=O)(O)O is clearly phosphonate. Just poorly processed. [PH](=O)(O)O ->
            # [PH0](=O)(O)O
            ("[*:1][PH:2](=O)(O)O", "[*:1][PH0:2](=O)(O)O"),
        
            # Terminal geminal diols are going to be carboxylates.
            ("[C;H1:1]([O;$([O-,OH])])([O;$([O-,OH])])", "[C:1](=O)(O)"),

            # Same with terminal geminol thiols
            ("[C;H1:1]([S;$([S-,SH])])([S;$([S-,SH])])", "[C:1](=S)(S)"),

            # Sometimes needless carbocations.
            ("[C+1;X3;v3:1]", "[CH+0:1]"),
            
            # Sometimes nucleobases are written with all single bonds. Not sure
            # why.
            ("[N:1]2CNC3C([NH2])NCNC32", "[n:1]2cnc3c([NH2])ncnc32"),  # adenine
            ("[N:1]1CNC(C1[NH]C(N)N2)C2~O", "[n:1]1cnc(c1[nH]c(N)n2)c2=O"),  # guanosine
            ("CC1C[N:1]C(NC1~O)~O", "CC1=C[N:1]C(NC1=O)=O"), # 5-methyluridine
            ("O~C1NC([N:1]CC1)~O", "O=C1NC([N:1]C=C1)=O"), # uridine
            ("O~C1NC(CC[N:1]1)N", "O=C1N=C(C=C[N:1]1)N") # cytidine
        ]
        
        # print("DEBUG!!!!DEBUG!!!!")
        # changed = False

        for patt1, patt2 in replacements:
            try:
                # Build the reaction SMARTS string (reactant >> product)
                reaction_smarts = patt1 + ">>" + patt2
                rxn = AllChem.ReactionFromSmarts(reaction_smarts)
                product_sets = rxn.RunReactants((rdmol,))
                if product_sets:
                    # Use the first set of products; note that many reactions yield tuples of products.
                    rdmol = product_sets[0][0]
                    # Update properties after reaction
                    for atom in rdmol.GetAtoms():
                        atom.UpdatePropertyCache(strict=False)
                    # changed = True
            except Exception as e:
                print(f"CAUGHT EXCEPTION: Could not process replacement: {patt1} >> {patt2} >> ", e)
                continue

        Chem.SanitizeMol(rdmol)

        # print("DEBUG!!!!DEBUG!!!!")
        # if changed:
        #     smiles2 = Chem.MolToSmiles(rdmol)
        #     if smiles != smiles2:
        #         with open("changed_smi.smi", "a") as f:
        #             f.write(smiles + "\t" + Chem.MolToSmiles(rdmol) + "\n")

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

        if isinstance(smiles_or_mol, str):
            return Chem.MolToSmiles(
                rdmol,
                isomericSmiles=False,  # No chirality
                canonical=True,  # e.g., all benzenes are written as aromatic
            )
        else:
            return rdmol

    except Exception as e:
        if none_if_fails:
            return None
        else:
            smiles = smiles_or_mol if isinstance(smiles_or_mol, str) else Chem.MolToSmiles(smiles_or_mol)
            print(f"CAUGHT EXCEPTION: Could not standardize SMILES: {smiles} >> ", e)
            # append to moose.log
            with open("moose.log", "a") as f:
                f.write(smiles + "\t" + str(e) + "\n")
            return smiles
