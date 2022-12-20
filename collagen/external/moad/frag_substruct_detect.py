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
    ]
]

charged_substructs_to_ignore = [
    Chem.MolFromSmarts(smi)
    for smi in [
        # azides
        "N=[N+]=[N;X1]",
        "N=[N+]=[N;H1;X2]",
        # nitro
        # Actually, I think I'll count nitro groups as charged.
    ]
]


def is_aromatic(mol: Chem.Mol) -> bool:
    if mol.HasSubstructMatch(Chem.MolFromSmarts("a")):
        # This probably isn't necessary given the next check, but leaving it
        # here just in case.
        return True
    return any(atom.GetIsAromatic() for atom in mol.GetAtoms())


def is_charged(mol: Chem.Mol) -> bool:
    # This code adapted from Dimorphite-DL
    if any(
        mol.HasSubstructMatch(charged_substruct)
        for charged_substruct in charged_substructs
    ):
        return True

    # If no formal charge, then the molecule is not charged.
    if all(atom.GetFormalCharge() == 0 for atom in mol.GetAtoms()):
        return False

    # If here, then there is a charged atom. If there are no charged nitrogen
    # atoms, count this molecule as charged.
    if not any(
        mol.HasSubstructMatch(charged_nitrogen_substruct)
        for charged_nitrogen_substruct in charged_nitrogen_substructs
    ):
        return True

    # The molecule has a charged nitrogen atom. Does it one of the prohibited
    # substructures? If not, count it as charged.
    if not any(
        mol.HasSubstructMatch(charged_substruct_to_ignore)
        for charged_substruct_to_ignore in charged_substructs_to_ignore
    ):
        return True

    # If here, there is a prohibited substructure. You must replace it with a
    # carbon and reassess the charge.
    for charged_substruct_to_ignore in charged_substructs_to_ignore:
        if mol.HasSubstructMatch(charged_substruct_to_ignore):
            mol = Chem.ReplaceSubstructs(
                mol, charged_substruct_to_ignore, Chem.MolFromSmiles("C")
            )[0]
            if is_charged(mol):
                return True

    return False
