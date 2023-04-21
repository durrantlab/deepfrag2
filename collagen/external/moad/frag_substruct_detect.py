# Code to identify aromatic/aliphatic and charged/uncharged groups. Not perfect,
# but pretty good.

from rdkit import Chem

charged_nitrogen_substructs = [
    Chem.MolFromSmarts(smi)
    for smi in [
        "[N+]",
        "[n+]",
    ]
]

# Get the charged moieties. Note that if there is an explicit charge, it will be
# caught elsewhere. This is only to find the moieties that are fully protonated
# but could be charged if they lost or gained a hydrogen atom.
charged_substructs = [
    Chem.MolFromSmarts(smi)
    for smi in [
        # Protonated carboxylate
        "[*]C(=O)[O;H1]",
        # Terminal amines, two hydrogens
        "[X4]-[N;H2;X3]",
        # Secondary amines, one hydrogen. But note that this doesn't catch
        # secondary amines immediately adjacent to the connection point, because
        # can't guarantee that would be sp3 hybridized.
        "[X4]-[N;H1;X3]-[X4]",
        # Tertiary amines, no hydrogens
        "[X4]-[N;H0;X3](-[X4])-[X4]",
        # Guanidines
        "[N]-[C](-[N])=[NX2]-[H]",
        "[C](-[N])=[NX2+0]",
        # Sulfur-related
        "[SX4](=O)(=O)(O-[C,c,N,n])-[OX2]-[H]",
        "[SX4](=O)(=O)(-[C,c,N,n])-[OX2]-[H]",
        "[SX3](=O)-[O]-[H]",
        # Phosphorus-related
        "[PX4](=O)(-[OX2]-[H])(-[O+0])-[OX2]-[H]",
        "[PX4](=O)(-[OX2]-[H])(-[C,c,N,n])-[OX2]-[H]",
        "[PX4](=O)(-[C,c,N,n,F,Cl,Br,I])(-[C,c,N,n,F,Cl,Br,I])-[OX2]-[H]",
        # Not sure below work (couldn't find examples).
        "[PX4](=O)(-[OX2]-[C,c,N,n,F,Cl,Br,I])(-[C,c,N,n,F,Cl,Br,I])-[OX2]-[H]",
        "[PX4](=O)(-[OX2]-[C,c,N,n,F,Cl,Br,I])(-[O+0]-[C,c,N,n,F,Cl,Br,I])-[OX2]-[H]",
        "[$([PX4:1]([OX2][C,c,N,n])(=O)([OX2][PX4](=O)([OX2])(O[H])))]O-[H]",
        "[$([PX4](=O)([OX2][PX4](=O)([OX2])(O[H]))([OX2][PX4](=O)(O[H])([OX2])))]O-[H]",
        # thioic acid
        "[C,c,N,n](=[O,S])-[SX2,OX2]-[H]",
        "[c,n]-[SX2]-[H]",

        # New ones adding for DeepFrag 
        # 
        # Aromatic nitrogens can often be positively charged near neutral pH.
        # Count aromatic nitrogens that are bound to two atoms, neither of which
        # is a hydrogen, as potentially positive.
        "[n;H0;D2]",

        # Above doesn't catch some sulfur-containing groups.
        "S(=O)[O;H1]",

        # To get, for exmaple, *OP(=S)(O[H])O[H]
        "P(=S)[O;H1]"
        "P(=O)[O;H1]"
        "P(=O)[S;H1]"
        "P(=S)[S;H1]"
    ]
]

charged_substructs_to_ignore = [
    (Chem.MolFromSmarts(smi[0]), Chem.MolFromSmarts(smi[1]))
    for smi in [
        # azides
        ("N=[N+]=[N;X1]", "C"),
        ("N=[N+]=[N;H1;X2]", "C"),
        # nitro
        # Actually, I think I'll count nitro groups as charged.
        # Sulfonamide-like
        ("S(=O)N", "S(=O)C"),
        ("P(=O)N", "P(=O)C")
    ]
]


def is_aromatic(mol: Chem.Mol) -> bool:
    # Below is overkill. Could probably just keep the first one.
    return (
        len(mol.GetAromaticAtoms()) > 0
        or mol.HasSubstructMatch(Chem.MolFromSmarts("a"))
        or any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    )


def is_charged(mol: Chem.Mol) -> bool:
    # If formal charge on any atom, then the molecule is charged. Easy solution.
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            # print("1", "formal charge found", Chem.MolToSmiles(mol))
            return True

    # if all(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
    #     return True

    # If you get here, you need to consider the possibility that it could be
    # ionizable, even though neutral in SMILES.

    # Make copy of mol, so substitution doesn't change original
    mol = Chem.Mol(mol)

    # Go through and mask out certain nitrogens. There could be a prohibited
    # substructure that appears to be a charged nitrogen (e.g., azide). You must
    # replace it with a carbon.
    for charged_substruct_to_ignore, replacement in charged_substructs_to_ignore:
        if mol.HasSubstructMatch(charged_substruct_to_ignore):
            mol = Chem.ReplaceSubstructs(
                mol, charged_substruct_to_ignore, replacement
            )[0]

            # fix
            mol.UpdatePropertyCache(strict=False)

    for charged_substruct in charged_substructs:
        if mol.HasSubstructMatch(charged_substruct):
            # print("2", "charged substruct found", Chem.MolToSmiles(mol), Chem.MolToSmarts(charged_substruct))
            return True

    # print("3", "no charged substruct found", Chem.MolToSmiles(mol))
    return False

    # if any(
    #     mol.HasSubstructMatch(charged_substruct)
    #     for charged_substruct in charged_substructs
    # ):
    #     return True


    # If here, then there is a charged atom. If there are no charged nitrogen
    # atoms, count this molecule as charged. Charged nitrogen atoms are handled
    # separately below.
    # if not any(
    #     mol.HasSubstructMatch(charged_nitrogen_substruct)
    #     for charged_nitrogen_substruct in charged_nitrogen_substructs
    # ):
    #     return True

    # The molecule has a charged nitrogen atom. Does it one of the prohibited
    # substructures (e.g., azide)? If not, count it as charged.
    # if not any(
    #     mol.HasSubstructMatch(charged_substruct_to_ignore)
    #     for charged_substruct_to_ignore in charged_substructs_to_ignore
    # ):
    #     return True

    # If here, there is a prohibited substructure that appears to be a charged
    # nitrogen (e.g., azide). You must replace it with a carbon and reassess the
    # charge.
    # for charged_substruct_to_ignore in charged_substructs_to_ignore:
    #     if mol.HasSubstructMatch(charged_substruct_to_ignore):
    #         mol = Chem.ReplaceSubstructs(
    #             mol, charged_substruct_to_ignore, Chem.MolFromSmiles("C")
    #         )[0]
    #         if is_charged(mol):
    #             return True

    # # If you get here, it's not charged.
    # return False
