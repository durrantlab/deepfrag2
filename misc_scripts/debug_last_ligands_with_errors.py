import csv
import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Geometry import Point3D


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

    with open(path_csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            pdb_name = row[col_pdb_name]
            sdf_name = row[col_sdf_name]
            if os.path.exists(path_pdb_sdf_files + os.sep + pdb_name) and os.path.exists(path_pdb_sdf_files + os.sep + sdf_name):
                read_mol(sdf_name, path_pdb_sdf_files, row[col_parent_smi], row[col_first_frag_smi], row[col_second_frag_smi], row[col_first_ligand_template], row[col_second_ligand_template])


def parent_smarts_to_mol(smi):
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


def read_mol(sdf_name, path_pdb_sdf_files, parent_smi, first_frag_smi, second_frag_smi, first_ligand_template,
             second_ligand_template):
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

            first_frag_smi = remove_mult_bonds_by_smi_to_smi(first_frag_smi)
            first_frag_smi = parent_smarts_to_mol(first_frag_smi)

            second_frag_smi = remove_mult_bonds_by_smi_to_smi(second_frag_smi)
            second_frag_smi = parent_smarts_to_mol(second_frag_smi)

            if conf and connect_coord and first_frag_smi and second_frag_smi:
                print("OK")

    else:
        return None, None, None


if __name__ == "__main__":
    read_data_from_csv("/home/garcc116/share_data/Anonymized_MMP/data/Full_Matched_Molecular_Pairs_Data.txt,/home/garcc116/share_data/Anonymized_MMP/data/csdb_structures,protein_pdb,ligand_pdb,FirstParent,FirstFragment,SecondFragment,Endpoint_First,Endpoint_Second,FirstSmiles,SecondSmiles")
