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


def read_mol(sdf_name, path_pdb_sdf_files, parent_smi, first_frag_smi, second_frag_smi, first_ligand_template, second_ligand_template):
    path_to_mol = path_pdb_sdf_files + os.sep + sdf_name
    if sdf_name.endswith(".pdb"):
        # read ligand from PDB file and assign bonds according to first smiles
        try:
            ref_mol = AllChem.MolFromPDBFile(path_to_mol, removeHs=False)
            template = Chem.MolFromSmiles(first_ligand_template)
            ref_mol = AllChem.AssignBondOrdersFromTemplate(template, ref_mol)
        except:
            ref_mol = None

        # try to assign bonds according to second smiles
        # seem to be that the second smiles is never useful to assign bonds
        if not ref_mol:
            try:
                ref_mol = AllChem.MolFromPDBFile(path_to_mol, removeHs=False)
                template = Chem.MolFromSmiles(second_ligand_template)
                ref_mol = AllChem.AssignBondOrdersFromTemplate(template, ref_mol)
            except:
                ref_mol = None

        # Use ligand into PDB file with single bonds only
        if not ref_mol:
            try:
                c_parent_smi = remove_mult_bonds_by_smi_to_smi(parent_smi)
                c_parent_smi = parent_smarts_to_mol(c_parent_smi)

                c_first_frag_smi = remove_mult_bonds_by_smi_to_smi(first_frag_smi)
                c_first_frag_smi = parent_smarts_to_mol(c_first_frag_smi)
                c_second_frag_smi = remove_mult_bonds_by_smi_to_smi(second_frag_smi)
                c_second_frag_smi = parent_smarts_to_mol(c_second_frag_smi)

                ref_mol = AllChem.MolFromPDBFile(path_to_mol, removeHs=False)
                ref_mol = remove_mult_bonds(ref_mol)

                parent_smi = c_parent_smi
                first_frag_smi = c_first_frag_smi
                second_frag_smi = c_second_frag_smi
            except:
                return None, None, None

    Chem.RemoveAllHs(ref_mol)
    # it is needed to get 3D coordinates for parent to voxelize.
    r_parent = get_sub_mol(ref_mol, parent_smi, sdf_name, None, None)
    r_first_frag_smi = get_sub_mol(ref_mol, second_frag_smi, sdf_name, None, None)

    if r_parent and r_first_frag_smi:
        print("OK")


# mol must be RWMol object
# based on https://github.com/wengong-jin/hgraph2graph/blob/master/hgraph/chemutils.py
def get_sub_mol(mol, smi_sub_mol, sdf_name, log_for_fragments, log_for_3d_coordinates, get_sub_structure=True):
    # getting substructure
    if isinstance(smi_sub_mol, str):
        patt = Chem.MolFromSmarts(smi_sub_mol)
        mol_smile = Chem.MolToSmiles(patt)
        mol_smile = mol_smile.replace("*/", "[H]").replace("*", "[H]")
        patt_mol = Chem.MolFromSmiles(mol_smile)
        Chem.RemoveAllHs(patt_mol)
        if not get_sub_structure:
            return patt_mol
    else:
        patt = smi_sub_mol
        patt_mol = smi_sub_mol

    sub_atoms = mol.GetSubstructMatch(patt_mol, useChirality=False, useQueryQueryMatches=False)
    if len(sub_atoms) == 0:
        return None

    # it is created the mol object for the obtained substructure
    mol.UpdatePropertyCache(strict=True)
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    # it is added the bonds corresponding to the obtained substructure
    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms:
                continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx():  # each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    # clean molecule
    try:
        new_mol = new_mol.GetMol()
        rdmolops.Cleanup(new_mol)
        rdmolops.RemoveStereochemistry(new_mol)
    except Exception as e:
        return None

    # assign 3D coordinates
    for i in [0, 1]:
        try:
            # new_mol = new_mol.GetMol()
            new_mol.UpdatePropertyCache(strict=True)
            new_mol = Chem.MolToMolBlock(new_mol)
            new_mol = Chem.MolFromMolBlock(new_mol)
            conf = new_mol.GetConformer()
            for idx in sub_atoms:
                a = mol.GetAtomWithIdx(idx)
                x, y, z = mol.GetConformer().GetAtomPosition(idx)
                conf.SetAtomPosition(atom_map[a.GetIdx()], Point3D(x, y, z))
            break
        except:
            if i == 1:
                return None
            else:
                mol_smile = Chem.MolToSmiles(new_mol)
                mol_smile = mol_smile.upper().replace("CL", "Cl").replace("BR", "Br")
                new_mol = Chem.MolFromSmiles(mol_smile)

    # find out the connector atom
    # NOTE: according to several runs, the connector atom is always allocated in the position 0 into the recovered substructure ('new_mol' variable)
    # but this was implemented just in case the connector atom is in a position other than 0.
    # this implementation has linear complexity and it is so fast
    if isinstance(smi_sub_mol, str):
        for idx in sub_atoms:
            a = mol.GetAtomWithIdx(idx)
            if patt.GetAtomWithIdx(atom_map[a.GetIdx()]).GetSymbol() == "*":  # this is the connector atom
                new_mol.GetAtomWithIdx(atom_map[a.GetIdx()]).SetAtomicNum(0)
                break
    else:
        # Get the connection point and add it to the data row
        for atom in new_mol.GetAtoms():
            if atom.HasProp("was_dummy_connected") and atom.GetProp("was_dummy_connected") == "yes":
                new_mol.GetAtomWithIdx(atom_map[a.GetIdx()]).SetAtomicNum(0)
                break

    Chem.RemoveAllHs(new_mol)
    return new_mol


if __name__ == "__main__":
    read_data_from_csv("/home/garcc116/share_data/Anonymized_MMP/data/Full_Matched_Molecular_Pairs_Data.txt,/home/garcc116/share_data/Anonymized_MMP/data/csdb_structures,protein_pdb,ligand_pdb,FirstParent,FirstFragment,SecondFragment,Endpoint_First,Endpoint_Second,FirstSmiles,SecondSmiles")
