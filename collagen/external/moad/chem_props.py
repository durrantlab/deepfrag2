"""Code to roughly identify aromatic/aliphatic and acid/base/neutral groups.
"""

from rdkit import Chem

acid_substructs_smi = [
    # Carboxylates
    "[*]C(=O)[O+0;H1]",
    "[*]C(=O)[O-]",
    # Hydroxyl groups bound to aromatic rings
    "c[O+0;H1]",
    "c[O-]",
    # Thiol groups bound to aromatic rings
    "c[S+0;H1]",
    "c[S-]",
    # Phosphate-like
    "P(=O)[O+0;H1]",
    "P(=O)[O-]",
    # Sulfate-like
    "S(=O)[O+0;H1]",
    "S(=O)[O-]",
    # Tetrazoles
    "c1nnn[nH]1",
    "c1nnn[n-]1",
    "c1nn[nH]n1",
    "c1nn[n-]n1",
    # sulfonamides are acidic. NOTE: for NS(=O)(=O)c1ccccc1, predicted pKa
    # is 10.24. For CS(N)(=O)=O, it is 13.11. For O=S1(=O)CCCCN1, it is
    # 12.55. For O=S1(=O)NC=CC=C1, it is 8.52. Not going to do all
    # sulfonamides, just ones attached to aromatic rings.
    "O=S(c)[N+0;H1;X3]",  # Covers CNS(=O)(=O)c1ccccc1
    "O=S(c)[N+0;H2;X3]",  # Covers NS(=O)(=O)c1ccccc1
    "O=S(c)[N-;H0;X2]",  # Covers C[N-]S(=O)(=O)c1ccccc1
    "O=S(c)[N-;H1;X2]",  # Covers [NH-]S(=O)(=O)c1ccccc1
    "O=S[N+0;H1;X3](c)",  # Covers aromatic off nitrogen, e.g., CS(=O)(=O)Nc1ccccc1
    "O=S[N+0;H2;X3](c)",  # Covers aromatic off nitrogen, e.g., CS(=O)(=O)Nc1ccccc1
    "O=S[N-;H0;X2](c)",  # Covers aromatic off nitrogen, e.g., CS(=O)(=O)[N-]c1ccccc1
    ### DON'T USE THE BELOW ###
    # Thiols have pretty low pKa, much lower than alcohols. But not used
    # much in drugs.
    # "C[S-]",
    # "C[S;H1]",
    # Nitro groups aren't really acidic.
    # "[N+](=O)[O-]",
    # "[N+](=O)[O;H1]",
    # Could consider N[O-] and n[O-] to be acidic, but they are pretty rare
    # substructures.
]

base_substructs_smi = [
    # Primary amines. For simplicity's sake, must be bound to SP3 carbon (to
    # avoid amide).
    "[C;X4][N+0;H2;X3]",
    "[C;X4][N+;H3;X4]",
    # Secondary amines. Also covers Piperazine, Piperidine, Pyrrolidine,
    # Aziridines. For simplicity's sake, must be bound to SP3 carbons (to
    # avoid sulfonamide).
    "[C;X4][N+0;H1;X3][C;X4]",
    "[C;X4][N+;H2;X4][C;X4]",
    # Tertiary amines. For simplicity's sake, must be bound to SP3 carbons.
    "[C;X4][N+0;H0;X3]([C;X4])[C;X4]",
    "[C;X4][N+;H1;X4]([C;X4])[C;X4]",
    # Imine-like
    "[N+0;H0;X2]",
    "[N+;H1;X3]",
    "[N+0;H1;X2]",
    "[N+;H2;X3]",
]

acid_substructs = [Chem.MolFromSmarts(smi) for smi in acid_substructs_smi]
base_substructs = [Chem.MolFromSmarts(smi) for smi in base_substructs_smi]

# TODO: Neutral can just be not acid, not base.

# TODO: What percentage are acids and bases? If small, maybe just use general
# model for neutral.

# TODO: Make sure lookup-table also matches properties of training/testing.

# TODO: Send all domain-specific models to Shayne when ready.

# TODO: Look at actually used. Lots of repeated, you ask for acids but not
# always acids. Not sure what to make of it. Need code review.


def is_aromatic(mol: Chem.Mol) -> bool:
    """Determine if a molecule is aromatic. Below is overkill. Could probably
    just keep the first one.
    
    Args:
        mol: RDKit molecule object.
        
    Returns:
        True if aromatic, False if not.
    """
    return (
        len(mol.GetAromaticAtoms()) > 0
        or mol.HasSubstructMatch(Chem.MolFromSmarts("a"))
        or any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    )


def is_acid(mol: Chem.Mol, testing=False) -> bool:
    """Determine if a molecule is an acid.

    Args:
        mol: RDKit molecule object.
        testing: If True, return tuple of (True/False, substructure matched). If
            False, return only True/False.

    Returns:
        True if an acid, False if not.
    """

    # Make copy of mol, so substitution doesn't change original
    mol = Chem.Mol(mol)

    if testing:
        return next(
            (
                (True, acid_substructs_smi[i])
                for i, acid_substruct in enumerate(acid_substructs)
                if mol.HasSubstructMatch(acid_substruct)
            ),
            (False, None),
        )

    return any(
        mol.HasSubstructMatch(acid_substruct) for acid_substruct in acid_substructs
    )


def is_base(mol: Chem.Mol, testing=False) -> bool:
    """Determine if a molecule is a base.

    Args:
        mol: RDKit molecule object.
        testing: If True, return tuple of (True/False, substructure matched). If
            False, return only True/False.

    Returns:
        True if a base, False if not.
    """

    # Make copy of mol, so substitution doesn't change original
    mol = Chem.Mol(mol)

    if testing:

        return next(
            (
                (True, base_substructs_smi[i])
                for i, base_substruct in enumerate(base_substructs)
                if mol.HasSubstructMatch(base_substruct)
            ),
            (False, None),
        )

    return any(
        mol.HasSubstructMatch(base_substruct) for base_substruct in base_substructs
    )


def is_neutral(mol: Chem.Mol) -> bool:
    """Determine if a molecule is neutral. If not acid and not base, assume
    neutral.

    Args:
        mol: RDKit molecule object.

    Returns:
        True if a neutral, False if not.
    """

    return not is_acid(mol) and not is_base(mol)


if __name__ == "__main__":
    import pandas as pd
    import sys

    # If no arguments, use filename "chem_props_test.smi"
    filename = "chem_props_test.smi" if len(sys.argv) == 1 else sys.argv[1]

    smis = []
    names = []
    types = []
    mols = []
    substruct_matches = []
    with open("chem_props_test.smi", "r") as f:
        for line in f:
            smi, name = line.strip().split()
            name, type = name.split("__")
            mol = Chem.MolFromSmiles(smi)
            smis.append(smi)
            names.append(name)
            types.append(type)
            mols.append(mol)

    # Create empty dataframe.
    df = pd.DataFrame(columns=["smiles", "name"])
    df["smiles"] = smis
    df["name"] = names
    df["correct_type"] = types

    acid_cats = [is_acid(mol, True) for mol in mols]
    base_cats = [is_base(mol, True) for mol in mols]

    df["predict_acid"] = [e[0] for e in acid_cats]
    df["acid_match"] = [e[1] for e in acid_cats]
    df["predict_base"] = [e[0] for e in base_cats]
    df["base_match"] = [e[1] for e in base_cats]
    df["predict_aromatic"] = [is_aromatic(mol) for mol in mols]

    print(df)

    # Save to tsv.
    df.to_csv("chem_props_test.tsv", sep="\t", index=False)

