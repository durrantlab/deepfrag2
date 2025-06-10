import os
from rdkit import Chem
from typing import List, Tuple
from collagen.core.molecules.mol import Mol
from rdkit.Chem.Descriptors import ExactMolWt
import argparse


# Method 'split_bonds' implemented in the BackedMol class in mol.py
def split_bonds(
        rdmol, only_single_bonds: bool = True, max_frag_size: int = -1
) -> List[Tuple["Mol", "Mol"]]:
    num_mols = len(Chem.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False))
    assert (
            num_mols == 1
    ), f"Error, calling split_bonds() on a Mol with {num_mols} parts."

    pairs = []
    for i in range(rdmol.GetNumBonds()):

        # Filter single bonds.
        if (
                only_single_bonds
                and rdmol.GetBondWithIdx(i).GetBondType()
                != Chem.rdchem.BondType.SINGLE
        ):
            continue

        split_mol = Chem.rdmolops.FragmentOnBonds(rdmol, [i])
        fragments = Chem.GetMolFrags(split_mol, asMols=True, sanitizeFrags=False)

        # Skip if this did not break the molecule into two pieces.
        if len(fragments) != 2:
            continue

        parent = from_rdkit(fragments[0])
        frag = from_rdkit(fragments[1])

        if mass(parent) < mass(frag):
            frag, parent = parent, frag

        # Ensure the fragment has at least one heavy atom.
        if frag.GetNumHeavyAtoms() == 0:
            continue

        # Filter by atomic mass (if enabled).
        if max_frag_size != -1 and mass(frag) > max_frag_size:
            continue

        pairs.append((parent, frag))

    return pairs


def mass(rdmol) -> float:
    return ExactMolWt(rdmol)


def from_rdkit(rdmol: "rdkit.Chem.rdchem.Mol", strict: bool = True) -> "BackedMol":
    rdmol.UpdatePropertyCache(strict=strict)
    return rdmol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_sdf_files', type=str, required=True, help='Path to SDF files')
    args = parser.parse_args()

    all_files = os.listdir(args.path_to_sdf_files)
    sdf_files = [
        os.path.abspath(os.path.join(args.path_to_sdf_files, f)) for f in all_files if f.endswith(".sdf")
    ]

    all_fragments = []
    for sdf_file in sdf_files:
        r = Chem.SDMolSupplier(sdf_file)
        for ligand in r:
            pairs = split_bonds(ligand)
            for _, frag in pairs:
                smi = Chem.MolToSmiles(frag)
                smi = '[*]' + smi.split('*]')[1]
                all_fragments.append(smi)
        r.reset()

    all_fragments = set(all_fragments)

    all_fragments_file = args.path_to_sdf_files + os.sep + "all_fragments.smiles"
    with open(all_fragments_file, "w") as file:
        for frag in all_fragments:
            file.write(frag)
            file.write('\n')
        file.close()


if __name__ == "__main__":
    main()
