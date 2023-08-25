from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import EditableMol
import tqdm
from tqdm.contrib.concurrent import process_map
import numpy as np

import json
from multiprocessing import Pool, cpu_count

import sys


def merge_smiles(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES provided.")

    # Find the atom indices of the asterisks more efficiently
    idx1, idx2 = None, None
    for atom in mol1.GetAtoms():
        if atom.GetSymbol() == "*":
            idx1 = atom.GetIdx()
            break
    for atom in mol2.GetAtoms():
        if atom.GetSymbol() == "*":
            idx2 = atom.GetIdx()
            break

    # The rest remains largely the same...
    idx1_neighbor = [
        atom.GetIdx() for atom in mol1.GetAtomWithIdx(idx1).GetNeighbors()
    ][0]
    idx2_neighbor = [
        atom.GetIdx() for atom in mol2.GetAtomWithIdx(idx2).GetNeighbors()
    ][0]

    emol = EditableMol(mol1)
    old_new_map = {atom.GetIdx(): emol.AddAtom(atom) for atom in mol2.GetAtoms()}

    for bond in mol2.GetBonds():
        emol.AddBond(
            old_new_map[bond.GetBeginAtomIdx()],
            old_new_map[bond.GetEndAtomIdx()],
            bond.GetBondType(),
        )

    emol.AddBond(idx1_neighbor, old_new_map[idx2_neighbor], Chem.rdchem.BondType.SINGLE)
    emol.RemoveAtom(old_new_map[idx2])
    emol.RemoveAtom(idx1)

    merged_mol = emol.GetMol()
    Chem.SanitizeMol(merged_mol)
    return Chem.MolToSmiles(merged_mol)


# 1. Create a worker function
def process_entry(entry):
    groundTruth = "groundTruth" if "groundTruth" in entry else "correct"
    receptor = entry[groundTruth]["receptor"]
    parent = entry[groundTruth]["parentSmiles"]

    lig_data_entry = {}

    if receptor not in lig_data_entry:
        correct_frag = entry[groundTruth]["fragmentSmiles"]
        correct_ligand = merge_smiles(parent, correct_frag)

        lig_data_entry[receptor] = {"correct": correct_ligand, "predicted": []}

    predicted_frags = [
        s["smiles"] for s in entry["avgOfCheckpoints"]["closestFromLabelSet"]
    ]
    predicted_ligands = [merge_smiles(parent, frag) for frag in predicted_frags]

    lig_data_entry[receptor]["predicted"].append(
        {"predLigs": predicted_ligands, "predFrags": predicted_frags, "parent": parent}
    )

    return lig_data_entry


def main():
    with open(sys.argv[1], "r") as f:
        data = json.load(f)

    # Use process_map instead of Pool.map
    results = process_map(
        process_entry, data["entries"], max_workers=cpu_count(), chunksize=100
    )

    lig_data = {}

    for result in results:
        pdbid = [k for k in result.keys()][0]

        if pdbid not in lig_data:
            lig_data[pdbid] = result[pdbid]
            continue

        # Adding to existing.

        # NOTE: No need to update correct. Should be the same for all entries
        # associated with this pdb.

        lig_data[pdbid]["predicted"] += result[pdbid]["predicted"]

    # Collect all fragments used in the dataset
    all_frags = set()
    for pdbid in lig_data:
        for entry in lig_data[pdbid]["predicted"]:
            all_frags.update(entry["predFrags"])
        
        # also remove fragments from predicteds
        # lig_data[pdbid]["predicted"] = [p["predLigs"] for p in lig_data[pdbid]["predicted"]]
    all_frags = list(all_frags)

    # Now go through each ligand set and make decoys. Use tqdm for progress bar.
    for pdbid in tqdm.tqdm(lig_data):
        correct = lig_data[pdbid]["correct"]
        for idx, entry in enumerate(lig_data[pdbid]["predicted"]):
            num_decoys = len(entry["predLigs"])
            parent = entry["parent"]

            decoy_ligs = []

            while len(decoy_ligs) < num_decoys:
                # Randomly pick a fragment
                frag = np.random.choice(all_frags)

                # Make the mol
                decoy = merge_smiles(parent, frag)

                # Check if it's already in the list
                if decoy in decoy_ligs:
                    continue

                # Check if it's in the list of predicted ligands
                if decoy in entry["predLigs"]:
                    continue

                # Make sure it is not the correct ligand
                if decoy == correct:
                    continue

                # If it's not in either, add it to the list
                decoy_ligs.append(decoy)
            
            lig_data[pdbid]["predicted"][idx]["decoys"] = decoy_ligs
            
            # import pdb; pdb.set_trace()

    # Remove "parent" and "predFrags" from lig_data
    for pdbid in lig_data:
        for idx, entry in enumerate(lig_data[pdbid]["predicted"]):
            del lig_data[pdbid]["predicted"][idx]["parent"]
            del lig_data[pdbid]["predicted"][idx]["predFrags"]

    import pdb

    pdb.set_trace()



if __name__ == "__main__":
    main()
