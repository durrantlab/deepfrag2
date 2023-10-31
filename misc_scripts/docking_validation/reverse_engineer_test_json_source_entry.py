import json
import sys
import glob
import os
import re
from merge_smiles_from_test_json import mol_is_complicated

test_json_path = sys.argv[1]
docking_dir_root = sys.argv[2] + "/"

test_data = json.load(open(test_json_path))["entries"]

correct_key = "correct" if "correct" in test_data[0] else "groundTruth"

all_data = []

for pdb_dir in glob.glob(docking_dir_root + "????"):
    pdb_id = os.path.basename(pdb_dir)
    test_data_for_this_pdb_id = [d for d in test_data if d[correct_key]["receptor"] == pdb_id]
    cryst_lig_path = pdb_dir + "/*cryst_lig.smi"
    cryst_lig_paths = glob.glob(cryst_lig_path)
    if len(cryst_lig_paths) == 0:
        # Skip
        continue
    cryst_lig_path = cryst_lig_paths[0]
    with open(cryst_lig_path) as f:
        cryst_lig_smi = f.readline().strip()

    # TODO: Unfortunately, glob order does not match test_data_for_this_pdb_id order!!!!
    def sort_key(filename):
        match = re.search(r'batch(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    batch_dirs = sorted(glob.glob(pdb_dir + "/batch*"), key=sort_key)
    print(batch_dirs)
    for batch_idx, batch_dir in enumerate(batch_dirs):
        pred_ligs_path = batch_dir + "/pred_ligs.smi"
        with open(pred_ligs_path) as f:
            pred_lig_smis_from_dsk = [l.split()[0] for l in f.readlines()]

        decoy_ligs_path = batch_dir + "/decoys.smi"
        if os.path.exists(decoy_ligs_path):
            with open(decoy_ligs_path) as f:
                decoy_lig_smis_from_dsk = [l.split()[0] for l in f.readlines()]
        else:
            decoy_lig_smis_from_dsk = []

        test_data_for_this_batch = json.loads(json.dumps(test_data_for_this_pdb_id[batch_idx]))
        del test_data_for_this_batch[correct_key]["pcaProjection"]
        # pred_frags = test_data_for_this_batch["perCheckpoint"][0]["averagedPrediction"]["closestFromLabelSet"]
        # pred_frags = [e["smiles"] for e in pred_frags]
        # pred_frags = [frag for frag in pred_frags if not mol_is_complicated(frag)]
        # print(len(pred_lig_smis_from_dsk))
        # print(len(pred_frags))
        # print("====")
        # [
        #     {
        #         "ligand": pred_lig_smis_from_dsk[e_idx],
        #         "parent": test_data_for_this_batch[correct_key]["parentSmiles"],
        #         "frag": e
        #     }
        #     for e_idx, e in enumerate(pred_frags)
        # ]
        del test_data_for_this_batch["perCheckpoint"]
        del test_data_for_this_batch["avgOfCheckpoints"]
        test_data_for_this_batch["pdbid"] = test_data_for_this_batch[correct_key]["receptor"]
        test_data_for_this_batch["batch"] = os.path.basename(batch_dir) 
        test_data_for_this_batch["cryst_ref"] = {
            "ligand": cryst_lig_smi,
            "parent": test_data_for_this_batch[correct_key]["parentSmiles"],
            "frag": test_data_for_this_batch[correct_key]["fragmentSmiles"],
            "connection": test_data_for_this_batch[correct_key]["connectionPoint"]
        }
        del test_data_for_this_batch[correct_key]
        test_data_for_this_batch["pred_ligands"] = pred_lig_smis_from_dsk
        test_data_for_this_batch["decoy_ligands"] = decoy_lig_smis_from_dsk
        all_data.append(test_data_for_this_batch)
        # print(json.dumps(test_data_for_this_batch, indent=2))

        # print(pdb_id, batch_dir, cryst_lig_smi, pred_lig_smis, json.dumps(test_data_for_this_batch, indent=2))

with open("docking_prep_data.json", "w") as f:
    json.dump(all_data, f, indent=2)