from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import EditableMol
import tqdm
from tqdm.contrib.concurrent import process_map
import numpy as np
import os
from os.path import dirname, exists, abspath
from os import system
import requests
from glob import glob
import argparse
import hashlib

import json
from multiprocessing import Pool, cpu_count

import sys

DEBUG = True
OPENBABEL_EXEC = "/Users/jdurrant/opt/anaconda3/bin/obabel"
SMINA_EXEC = "/Applications/smina/smina.osx"


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
    parent = entry[groundTruth]["parentSmiles"]
    if mol_is_complicated(parent):
        return {}

    correct_frag = entry[groundTruth]["fragmentSmiles"]
    correct_ligand = merge_smiles(parent, correct_frag)

    if mol_is_complicated(correct_ligand):
        return {}

    receptor = entry[groundTruth]["receptor"]

    lig_data_entry = {receptor: {"correct": correct_ligand, "predicted": []}}
    predicted_frags = [
        s["smiles"] for s in entry["avgOfCheckpoints"]["closestFromLabelSet"]
    ]

    # Remove chiral fragments
    predicted_frags = [frag for frag in predicted_frags if not mol_is_complicated(frag)]

    predicted_ligands = [merge_smiles(parent, frag) for frag in predicted_frags]

    lig_data_entry[receptor]["predicted"].append(
        {"predLigs": predicted_ligands, "predFrags": predicted_frags, "parent": parent}
    )

    return lig_data_entry


def mol_is_chiral(mol):
    """
    Checks if a molecule is chiral or not, even if it is not explicitly
    defined as such.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule to check.

    Returns:
        bool: True if the molecule is chiral, False otherwise.
    """

    # If the molecule is None, it is not chiral
    if mol is None:
        return False

    # If the molecule is chiral, it will have a non-zero number of chiral
    # centers.
    return len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) > 0


def mol_has_nonaromatic_ring(mol):
    """
    Checks if a molecule has a non-aromatic ring.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule to check.

    Returns:
        bool: True if the molecule has a non-aromatic ring, False otherwise.
    """

    # If the molecule is None
    if mol is None:
        return False

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    for ring in atom_rings:
        if not all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring):
            # print(f"Found non-aromatic ring in {Chem.MolToSmiles(mol)}")
            return True

    return False


# 5wfz
# smi = "O=C(NCCS(=O)(=O)c1ccccc1)c1nc(C2CCCN2C(=O)c2c(Cl)cncc2Cl)[nH]c(=O)c1O"
# mol = Chem.MolFromSmiles(smi)
# print(mol_has_nonaromatic_ring(mol))


def mol_is_complicated(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol_is_chiral(mol):
        return True
    if mol_has_nonaromatic_ring(mol):
        return True
    return False


def make_predicted_merged_mols(test_json_filename):
    with open(test_json_filename, "r") as f:
        data = json.load(f)

    # Use process_map instead of Pool.map
    entries = data["entries"]
    if DEBUG:
        num_entries = len(entries)
        entries = entries[:5000]
        print("DEBUG MODE: Only using first 5000 entries out of", num_entries)
    results = process_map(
        process_entry, entries, max_workers=cpu_count(), chunksize=100
    )

    lig_data = {}

    for result in results:
        keys = [k for k in result.keys()]
        if not keys:
            continue
        pdbid = keys[0]

        if pdbid not in lig_data:
            lig_data[pdbid] = result[pdbid]
            continue

        # Adding to existing.

        # NOTE: No need to update correct. Should be the same for all entries
        # associated with this pdb.

        lig_data[pdbid]["predicted"] += result[pdbid]["predicted"]
    return lig_data


def collect_all_fragments_used(lig_data):
    # Collect all fragments used in the dataset
    all_frags = set()
    for pdbid in lig_data:
        for entry in lig_data[pdbid]["predicted"]:
            all_frags.update(entry["predFrags"])

        # also remove fragments from predicteds
        # lig_data[pdbid]["predicted"] = [p["predLigs"] for p in lig_data[pdbid]["predicted"]]
    all_frags = list(all_frags)

    # Remove those fragments that are chiral
    return [frag for frag in all_frags if not mol_is_complicated(frag)]


def make_decoys(lig_data, all_frags):
    # Now go through each ligand set and make decoys. Use tqdm for progress bar.
    for pdbid in tqdm.tqdm(lig_data, desc="Making decoys"):
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

                # Don't add it if it is chiral
                if mol_is_complicated(decoy):
                    continue

                # If it's not in either, add it to the list
                decoy_ligs.append(decoy)

            lig_data[pdbid]["predicted"][idx]["decoys"] = decoy_ligs

            # import pdb; pdb.set_trace()


def get_all_lig_info():
    ligs_to_keep = set()
    lig_to_smiles = {}
    with open("ligs_to_keep.txt", "r") as f:
        for l in f:
            if "\t" not in l:
                continue
            prts = l.strip().split("\t")
            if len(prts) < 2:
                continue
            lig_id, smiles = prts
            ligs_to_keep.add(lig_id)
            lig_to_smiles[lig_id] = smiles

    return ligs_to_keep, lig_to_smiles


def ligand_smi_to_pdbqt(filename):
    system(
        f"{OPENBABEL_EXEC} -p 7.4 --gen3D -ismi {filename} -opdbqt -O {filename}.pdbqt"
    )


def multiline_ligand_smi_to_pdbqt(filename):
    d = dirname(filename)
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            smi, name = line.strip().split()
            cmd = f'{OPENBABEL_EXEC} -p 7.4 --gen3D -:"{smi}" -O "{d}/{name}.pdbqt"'
            system(cmd)


def receptor_pdb_to_pdbqt(filename):
    system(f"{OPENBABEL_EXEC} -p 7.4 -xr -ipdb {filename} -opdbqt -O {filename}.pdbqt")


def get_pdb_lig(payload):
    out_dir, pdbid, ligs_to_keep, lig_datum, lig_to_smiles, cache_dir = payload
    pdb_dir = f"{out_dir}/{pdbid}/"
    system(f"mkdir -p {pdb_dir}")

    # Download the pdb using requests
    pdb_file = f"{pdb_dir}/{pdbid}.pdb"
    if not exists(pdb_file):
        url = f"https://files.rcsb.org/download/{pdbid}.pdb"

        # Convert url to hash that can be a filename
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_filename = f"{cache_dir}/{url_hash}"

        if exists(cache_filename):
            # Get from cache
            print(f"Using cache for {pdbid}...")
            with open(cache_filename, "rb") as f:
                content = f.read()
        else:
            # Download from internet
            print(f"Downloading {pdbid}...")
            print(cache_filename)
            r = requests.get(url, allow_redirects=True)
            content = r.content

            # Save to cache
            with open(cache_filename, "wb") as f:
                f.write(content)

        # Save to pdb_file
        with open(pdb_file, "wb") as f:
            f.write(content)

    # Separate out ligand and receptor
    lig_pdb_file = f"{pdb_dir}/{pdbid}_cryst_lig.pdb"
    recep_pdb_file = f"{pdb_dir}/{pdbid}_cryst_recep.pdb"

    if not exists(lig_pdb_file) or not exists(recep_pdb_file):
        # Get all the ligands (HETATMs)
        with open(pdb_file, "r") as f:
            pdb_lines = f.readlines()
        hetatm_lines = [l for l in pdb_lines if l.startswith("HETATM")]
        lig_resname_chain_resnums = {l[17:26].strip() for l in hetatm_lines}
        uniq_ligs = {l[17:20] for l in hetatm_lines}

        # Remove ones that have only two letters (after trim)
        # uniq_ligs = [l.strip() for l in uniq_ligs if len(l.strip()) > 2]

        # Keep only ones in ligs_to_keep
        uniq_ligs = [l for l in uniq_ligs if l in ligs_to_keep]

        if not uniq_ligs:
            # print("No ligands found for", pdbid)
            print("REMEMBER: GET UPDATED EVERY.CSV")
            return 0

        correct_ligand_smi = lig_datum["correct"]
        uniq_ligs = [[lig_to_smiles[l], l] for l in uniq_ligs]

        # Counts
        correct_ligand_smi_upper = correct_ligand_smi.upper()
        num_C = correct_ligand_smi_upper.count("C")
        num_N = correct_ligand_smi_upper.count("N")
        num_O = correct_ligand_smi_upper.count("O")
        num_P = correct_ligand_smi_upper.count("P")

        # Keep ones with same counts
        matching_uniq_ligs = [
            l
            for l in uniq_ligs
            if l[0].upper().count("C") == num_C
            and l[0].upper().count("N") == num_N
            and l[0].upper().count("O") == num_O
            and l[0].upper().count("P") == num_P
        ]

        num_matching_uniq_ligs = len(matching_uniq_ligs)
        if num_matching_uniq_ligs != 1:
            # print(pdbid)
            # print(f"Correct smi: {correct_ligand_smi}")
            # print(f"smis from every: {uniq_lig_smis}")
            print("REMEMBER: GET UPDATED EVERY.CSV")
            return 0

        if num_matching_uniq_ligs > 1:
            # This should never happen.
            print(pdbid)
            print(f"Correct smi: {correct_ligand_smi}")
            print(f"smis from every: {uniq_ligs}")
            return 0

        # You might as well use the bindingmoad-catalogued SMILES string, which
        # I trust more than mine.
        correct_ligand_smi = matching_uniq_ligs[0][0]

        resname_of_lig = matching_uniq_ligs[0][1]

        # Get the first item in lig_resname_chain_resnums that starts with resname_of_lig
        matching_resname_chain_resnum = [
            l for l in lig_resname_chain_resnums if l.startswith(resname_of_lig)
        ][0]

        lig_pdb_lines = [l for l in pdb_lines if matching_resname_chain_resnum in l]
        rest_pdb_lines = [
            l for l in pdb_lines if matching_resname_chain_resnum not in l
        ]

        # Write the ligand to a file
        with open(lig_pdb_file, "w") as f:
            f.write("".join(lig_pdb_lines))

        # Write the rest to a file
        # NOTE: Purposefully not removing any water molecules.
        with open(recep_pdb_file, "w") as f:
            f.write("".join(rest_pdb_lines))
        receptor_pdb_to_pdbqt(recep_pdb_file)

        # Output the correct ligand to a file
        smi_file = f"{pdb_dir}/{pdbid}_cryst_lig.smi"
        with open(smi_file, "w") as f:
            f.write(correct_ligand_smi)
        ligand_smi_to_pdbqt(smi_file)

    for batch_idx, predicted_data in enumerate(lig_datum["predicted"]):
        batch_dir = f"{pdb_dir}/batch{batch_idx}/"
        if exists(batch_dir):
            # Already processed this batch, probably
            continue
        system(f"mkdir -p {batch_dir}")

        predLigs = [
            d + "\tpredicted" + str(i + 1)
            for i, d in enumerate(predicted_data["predLigs"])
        ]
        decoys = [
            d + "\tdecoy" + str(i + 1) for i, d in enumerate(predicted_data["decoys"])
        ]

        # Write the predicted ligands to a file
        smi_file = f"{batch_dir}/pred_ligs.smi"
        with open(smi_file, "w") as f:
            f.write("\n".join(predLigs))
        multiline_ligand_smi_to_pdbqt(smi_file)

        # Write the decoys to a file
        smi_file = f"{batch_dir}/decoys.smi"
        with open(smi_file, "w") as f:
            f.write("\n".join(decoys))
        multiline_ligand_smi_to_pdbqt(smi_file)

    return 1


def make_docking_cmds(out_dir):
    # Find all the "decoy*pdbqt" files (recursively) that do not have associated
    # pdbqt.out files. Use recursive glob
    lig_pdbqt_files = [
        f
        for f in glob(f"{out_dir}/**/decoy*.pdbqt", recursive=True)
        + glob(f"{out_dir}/**/predicted*.pdbqt", recursive=True)
        if not exists(f + "_out.pdbqt")
    ]
    lig_pdbqt_files = [
        (f, abspath(glob(f"{dirname(f)}/../*recep.pdb.pdbqt")[0]))
        for f in lig_pdbqt_files
    ]
    lig_pdbqt_files = [
        (l, r, r.replace("recep.pdb.pdbqt", "lig.pdb")) for l, r in lig_pdbqt_files
    ]

    cryst_lig_pdbqt_files = [
        f
        for f in glob(f"{out_dir}/**/*cryst*.pdbqt", recursive=True)
        if not exists(f + "_out.pdbqt")
    ]
    cryst_lig_pdbqt_files = [
        (f, abspath(glob(f"{dirname(f)}/*recep.pdb.pdbqt")[0]))
        for f in cryst_lig_pdbqt_files
    ]
    cryst_lig_pdbqt_files = [
        (l, r, r.replace("recep.pdb.pdbqt", "lig.pdb"))
        for l, r in cryst_lig_pdbqt_files
    ]

    for lig, recep, cryst in lig_pdbqt_files + cryst_lig_pdbqt_files:
        if "_out" in lig:
            continue
        if "_out" in recep:
            continue
        if "_out" in cryst:
            continue
        cmd = f"{SMINA_EXEC} --cpu 1 --receptor {recep} --ligand {lig} --autobox_ligand {cryst} --out {lig}_out.pdbqt --log {lig}.log"
        print(cmd)


def prepare_docking(lig_data, args):
    out_dir = args.out_dir
    system(f"mkdir -p {out_dir}/")

    cache_dir = args.cache_dir
    system(f"mkdir -p {cache_dir}/")

    ligs_to_keep, lig_to_smiles = get_all_lig_info()

    # for pdbid in tqdm.tqdm(lig_data, desc="Preparing docking"):
    #     get_pdb_lig(out_dir, pdbid, ligs_to_keep, lig_data[pdbid], lig_to_smiles)

    # Do same as above, now using multiple processors
    pdbids = [pdbid for pdbid in lig_data.keys()]
    data = [
        (out_dir, pdbid, ligs_to_keep, lig_data[pdbid], lig_to_smiles, cache_dir)
        for pdbid in pdbids
    ]
    # pool = Pool(processes=cpu_count())
    # results = pool.map(get_pdb_lig, data)

    # Do same as above, now using multiple processors with tqdm. Use process_map
    results = process_map(get_pdb_lig, data, max_workers=cpu_count(), chunksize=100)

    make_docking_cmds(out_dir)


def main(args):
    lig_data = make_predicted_merged_mols(args.test_json)
    all_frags = collect_all_fragments_used(lig_data)
    make_decoys(lig_data, all_frags)

    # Remove "parent" and "predFrags" from lig_data
    for pdbid in tqdm.tqdm(lig_data, desc="Removing unnecessary data"):
        lig_data[pdbid]["predicted"] = [
            p
            for p in lig_data[pdbid]["predicted"]
            if len(p["predLigs"]) > 0 and len(p["decoys"]) > 0
        ]
        for idx, entry in enumerate(lig_data[pdbid]["predicted"]):
            del lig_data[pdbid]["predicted"][idx]["parent"]
            del lig_data[pdbid]["predicted"][idx]["predFrags"]

    # Do a "sanity check" to make sure all molecules are non-chiral
    for pdbid in tqdm.tqdm(lig_data, desc="Sanity check"):
        correct = lig_data[pdbid]["correct"]
        if mol_is_complicated(correct):
            print(f"Correct ligand for {pdbid} is chiral: {correct}")
            assert False
        for entry in lig_data[pdbid]["predicted"]:
            mols_to_consider = entry["predLigs"] + entry["decoys"]
            for mol in mols_to_consider:
                if mol_is_complicated(mol):
                    print(f"Chiral molecule found: {mol}")
                    assert False

    prepare_docking(lig_data, args)


if __name__ == "__main__":
    # Make argeparse. First argument is test_json file, second argument is
    # output directory, third is cache directory.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test_json", type=str, help="Path to the test json file generated by collagen.",
    )
    parser.add_argument(
        "out_dir",
        type=str,
        help="Path to the output directory where the docking files will be stored.",
    )
    parser.add_argument(
        "cache_dir",
        type=str,
        help="Path to the cache directory where the docking files will be stored.",
    )
    args = parser.parse_args()

    main(args)
