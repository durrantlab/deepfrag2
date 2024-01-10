import torch
from rdkit import Chem
import os
from rdkit.Chem import AllChem
import numpy as np
import csv
import pickle
from torch import nn


class PairDataEntry:
    def __init__(self, pdb_name, sdf_name, parent, frag1, frag2, act1, act2):
        self.pdb_name = pdb_name
        self.sdf_name = sdf_name
        self.parent = parent
        self.frag1 = frag1
        self.frag2 = frag2
        self.act1 = act1
        self.act2 = act2


def parent_smarts_to_mol(smi):
    try:
        # It's not enough to just convert to mol with MolFromSmarts. Need to keep track of
        # connection point.
        smi = smi.replace("[R*]", "*")
        mol = Chem.MolFromSmiles(smi)

        # Find the dummy atom.
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = atom.GetNeighbors()
                if neighbors:
                    # Assume only one neighbor
                    neighbor = neighbors[0]
                    neighbor.SetProp("was_dummy_connected", "yes")
                    # Remove dummy atom
                    eds = Chem.EditableMol(mol)
                    eds.RemoveAtom(atom.GetIdx())
                    mol = eds.GetMol()

        # Now dummy atom removed, but connection marked.
        return mol
    except:
        return None


def remove_mult_bonds_by_smi_to_smi(smi):
    smi = smi.upper()
    smi = smi.replace("=", "")
    smi = smi.replace("#", "")
    smi = smi.replace("BR", "Br").replace("CL", "Cl")
    return smi


def remove_mult_bonds(mol):
    # mol = Chem.MolFromSmiles(smi)
    emol = Chem.EditableMol(mol)
    for bond in mol.GetBonds():
        emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        emol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)

    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    # mol=Chem.AddHs(mol)
    return mol


def substruct_with_coords(mol, substruct_mol, atom_indices):
    # Find matching substructure
    # atom_indices = mol.GetSubstructMatch(substruct_mol)

    # Get the conformer from mol
    conf = mol.GetConformer()

    # Create new mol
    new_mol = Chem.RWMol(substruct_mol)

    # Create conformer for new mol
    new_conf = Chem.Conformer(new_mol.GetNumAtoms())

    # Set the coordinates
    for idx, atom_idx in enumerate(atom_indices):
        new_conf.SetAtomPosition(idx, conf.GetAtomPosition(atom_idx))

    # Add new conf
    new_mol.AddConformer(new_conf)

    # Convert to mol
    new_mol = new_mol.GetMol()

    return new_mol


def read_mol(sdf_name, path_pdb_sdf_files, parent_smi, first_frag_smi, second_frag_smi, first_ligand_template, second_ligand_template):
    path_to_mol = path_pdb_sdf_files + os.sep + sdf_name
    if sdf_name.endswith(".pdb"):

        pdb_mol = AllChem.MolFromPDBFile(path_to_mol, removeHs=False)
        if pdb_mol is None:
            # In at least one case, the pdb_mol appears to be unparsable. Must skip.
            return None, None, None

        # Get parent mol too.
        first_parent = parent_smi

        # Note that it's important to use MolFromSmarts here, not MolFromSmiles
        parent_mol = parent_smarts_to_mol(first_parent)

        try:
            # Check if substructure match
            atom_indices = pdb_mol.GetSubstructMatch(parent_mol, useChirality=False, useQueryQueryMatches=False)
            atom_indices = None if len(atom_indices) == 0 else atom_indices
        except:
            atom_indices = None

        if atom_indices is None:
            # Previous attempt failed. Try converting everything into single bonds. For parent molecule,
            # do on level of smiles to avoid errors.
            parent_smi = remove_mult_bonds_by_smi_to_smi(parent_smi)
            parent_mol = parent_smarts_to_mol(parent_smi)

            # Try converting everything into single bonds in ligand.
            pdb_mol = remove_mult_bonds(pdb_mol)

            # Note: Not necessary to remove chirality given useChirality=False flag below.
            try:
                atom_indices = pdb_mol.GetSubstructMatch(parent_mol, useChirality=False, useQueryQueryMatches=False)
                atom_indices = None if len(atom_indices) == 0 else atom_indices
            except:
                atom_indices = None

        if atom_indices is not None and len(atom_indices) == parent_mol.GetNumAtoms():

            # Success in finding substructure. Make new mol of just substructure.
            new_mol = substruct_with_coords(pdb_mol, parent_mol, atom_indices)

            # Get the connection point and add it to the data row
            for atom in new_mol.GetAtoms():
                if atom.HasProp("was_dummy_connected") and atom.GetProp("was_dummy_connected") == "yes":
                    atom_idx = atom.GetIdx()
                    break

            conf = new_mol.GetConformer()
            connect_coord = conf.GetAtomPosition(atom_idx)
            connect_coord = np.array([connect_coord.x, connect_coord.y, connect_coord.z])

            backed_parent = new_mol

            first_frag_smi = remove_mult_bonds_by_smi_to_smi(first_frag_smi)
            first_frag_smi = parent_smarts_to_mol(first_frag_smi)
            backed_frag1 = first_frag_smi if first_frag_smi else None

            second_frag_smi = remove_mult_bonds_by_smi_to_smi(second_frag_smi)
            second_frag_smi = parent_smarts_to_mol(second_frag_smi)
            backed_frag2 = second_frag_smi if second_frag_smi else None

            return backed_parent, backed_frag1, backed_frag2

    return None, None, None


def read_data_from_csv(paired_data_csv):
    # reading input parameters
    paired_data_csv_sep = paired_data_csv.split(",")
    path_csv_file = paired_data_csv_sep[0]  # path to the csv (or tab) file
    path_pdb_sdf_files = paired_data_csv_sep[1]  # path containing the pdb and/or sdf files
    col_pdb_name = paired_data_csv_sep[2]  # pdb file name containing the receptor
    col_sdf_name = paired_data_csv_sep[3]  # sdf (or pdb) file name containing the ligand
    col_parent_smi = paired_data_csv_sep[4]  # SMILES string for the parent
    col_first_frag_smi = paired_data_csv_sep[5]  # SMILES string for the first fragment
    col_second_frag_smi = paired_data_csv_sep[6]  # SMILES string for the second fragment
    col_act_first_frag_smi = paired_data_csv_sep[7]  # activity for the first fragment
    col_act_second_frag_smi = paired_data_csv_sep[8]  # activity for the second fragment
    col_first_ligand_template = paired_data_csv_sep[9]  # first SMILES string for assigning bonds to the ligand
    col_second_ligand_template = paired_data_csv_sep[10]  # if fail the previous SMILES, second SMILES string for assigning bonds to the ligand
    col_prevalence = paired_data_csv_sep[11] if len(paired_data_csv_sep) == 12 else None  # prevalence value (optional). Default is 1 (even prevalence for the fragments)

    data = []
    with open(path_csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            pdb_name = row[col_pdb_name]
            sdf_name = row[col_sdf_name]
            if os.path.exists(path_pdb_sdf_files + os.sep + pdb_name) and os.path.exists(path_pdb_sdf_files + os.sep + sdf_name):
                backed_parent, backed_first_frag, backed_second_frag = read_mol(sdf_name,
                                                                                path_pdb_sdf_files,
                                                                                row[col_parent_smi],
                                                                                row[col_first_frag_smi],
                                                                                row[col_second_frag_smi],
                                                                                row[col_first_ligand_template],
                                                                                row[col_second_ligand_template])

                if backed_parent:
                    # getting the smiles for parent.
                    try:
                        parent_smi = Chem.MolToSmiles(backed_parent, isomericSmiles=True)
                    except:
                        continue

                    # getting the smiles for first fragment.
                    if backed_first_frag:
                        try:
                            first_frag_smi = Chem.MolToSmiles(backed_first_frag, isomericSmiles=True)
                        except:
                            first_frag_smi = None

                    # getting the smiles for second fragment.
                    if backed_second_frag:
                        try:
                            second_frag_smi = Chem.MolToSmiles(backed_second_frag, isomericSmiles=True)
                        except:
                            second_frag_smi = None

                    act_first_frag_smi = float(row[col_act_first_frag_smi]) if backed_first_frag else float(0)
                    act_second_frag_smi = float(row[col_act_second_frag_smi]) if backed_second_frag else float(0)
                    prevalence_receptor = float(row[col_prevalence]) if col_prevalence else 1

                    data.append(PairDataEntry(pdb_name, sdf_name, parent_smi, first_frag_smi, second_frag_smi, act_first_frag_smi, act_second_frag_smi))

    return data


# Closer to 1 means more similar, closer to 0 means more dissimilar.
_cos = nn.CosineSimilarity(dim=1, eps=1e-6)


if __name__ == "__main__":
    root = "path"
    paired_data_csv = "path"
    predicted_fps_file = "path"
    calculated_fps_file = "path"
    predicted_fps = {}
    calculated_fps = {}

    print("Loading predicted fingerprints")
    predicted_fps = torch.load(predicted_fps_file)

    print("Loading calculated fingerprints")
    calculated_fps = torch.load(calculated_fps_file)

    frag1_most_similar_higher_act = csv.writer(os.path.abspath(os.path.join(root, "frag1_most_similar_higher_act.csv")))
    frag1_most_similar_higher_act.writerow(["pdb", "ligand", "parent", "frag1", "frag2", "act1", "act2"])
    frag1_most_similar_lower_act = csv.writer(os.path.abspath(os.path.join(root, "frag1_most_similar_lower_act.csv")))
    frag1_most_similar_lower_act.writerow(["pdb", "ligand", "parent", "frag1", "frag2", "act1", "act2"])
    frag2_most_similar_higher_act = csv.writer(os.path.abspath(os.path.join(root, "frag2_most_similar_higher_act.csv")))
    frag2_most_similar_higher_act.writerow(["pdb", "ligand", "parent", "frag1", "frag2", "act1", "act2"])
    frag2_most_similar_lower_act = csv.writer(os.path.abspath(os.path.join(root, "frag2_most_similar_lower_act.csv")))
    frag2_most_similar_lower_act.writerow(["pdb", "ligand", "parent", "frag1", "frag2", "act1", "act2"])

    print("Reading paired data")
    data = read_data_from_csv(paired_data_csv)

    print("Performing analysis")
    for idx, entry in enumerate(data):
        key = entry.pdb_name + "_" + entry.sdf_name + "_" + entry.parent
        if key in predicted_fps.keys():
            recep_parent_fps = predicted_fps[key]

            if entry.frag1 and entry.frag2:
                frag1_fps = calculated_fps[entry.frag1]
                frag2_fps = calculated_fps[entry.frag2]

                sim_to_frag1 = _cos(recep_parent_fps, frag1_fps)
                sim_to_frag2 = _cos(recep_parent_fps, frag2_fps)

                writer = None
                if sim_to_frag1 > sim_to_frag2:
                    if entry.act1 > entry.act2:
                        writer = frag1_most_similar_higher_act
                    else:
                        writer = frag1_most_similar_lower_act
                else:
                    if entry.act1 < entry.act2:
                        writer = frag2_most_similar_higher_act
                    else:
                        writer = frag2_most_similar_lower_act

                writer.writerow([entry.pdb_name, entry.sdf_name, entry.parent, entry.frag1, entry.frag2, entry.act1, entry.act2])
