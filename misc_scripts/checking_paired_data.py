
import os
import logging
import csv
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdmolops
import sys


class PairedPdbSdfCsvInterface(object):
    pdb_files = []
    sdf_x_pdb = {}
    parent_x_sdf_x_pdb = {}
    frag_and_act_x_parent_x_sdf_x_pdb = {}
    backed_mol_x_parent = {}

    fail_match_FirstSmiles_PDBLigand = None
    fail_match_SecondSmiles_PDBLigand = None
    ligand_not_contain_parent = None
    ligand_not_contain_first_frag = None
    ligand_not_contain_second_frag = None
    error_getting_3d_coordinates_for_parent = None
    error_getting_3d_coordinates_for_first_frag = None
    error_getting_3d_coordinates_for_second_frag = None
    error_standardizing_smiles_for_parent = None
    error_standardizing_smiles_for_first_frag = None
    error_standardizing_smiles_for_second_frag = None
    finally_used = None

    def __init__(
            self,
    ):
        self._creating_logger_files()

    def _creating_logger_files(self):
        self.__setup_logger('log_one', os.getcwd() + os.sep + "01_fail_match_FirstSmiles_PDBLigand.log")
        self.__setup_logger('log_two', os.getcwd() + os.sep + "02_fail_match_SecondSmiles_PDBLigand.log")
        self.__setup_logger('log_three', os.getcwd() + os.sep + "03_ligand_not_contain_parent.log")
        self.__setup_logger('log_four', os.getcwd() + os.sep + "04_ligand_not_contain_first-frag.log")
        self.__setup_logger('log_five', os.getcwd() + os.sep + "05_ligand_not_contain_second-frag.log")
        self.__setup_logger('log_six', os.getcwd() + os.sep + "06_error_getting_3d_coordinates_for_parent.log")
        self.__setup_logger('log_seven', os.getcwd() + os.sep + "07_error_getting_3d_coordinates_for_first-frag.log")
        self.__setup_logger('log_eight', os.getcwd() + os.sep + "08_error_getting_3d_coordinates_for_second-frag.log")
        self.__setup_logger('log_nine', os.getcwd() + os.sep + "09_error_standardizing_smiles_for_parent.log")
        self.__setup_logger('log_ten', os.getcwd() + os.sep + "10_error_standardizing_smiles_for_first-frag.log")
        self.__setup_logger('log_eleven', os.getcwd() + os.sep + "11_error_standardizing_smiles_for_second-frag.log")
        self.__setup_logger('log_twelve', os.getcwd() + os.sep + "12_finally_used.log")

        self.fail_match_FirstSmiles_PDBLigand = logging.getLogger('log_one')
        self.fail_match_FirstSmiles_PDBLigand.propagate = False
        self.fail_match_SecondSmiles_PDBLigand = logging.getLogger('log_two')
        self.fail_match_SecondSmiles_PDBLigand.propagate = False
        self.ligand_not_contain_parent = logging.getLogger('log_three')
        self.ligand_not_contain_parent.propagate = False
        self.ligand_not_contain_first_frag = logging.getLogger('log_four')
        self.ligand_not_contain_first_frag.propagate = False
        self.ligand_not_contain_second_frag = logging.getLogger('log_five')
        self.ligand_not_contain_second_frag.propagate = False
        self.error_getting_3d_coordinates_for_parent = logging.getLogger('log_six')
        self.error_getting_3d_coordinates_for_parent.propagate = False
        self.error_getting_3d_coordinates_for_first_frag = logging.getLogger('log_seven')
        self.error_getting_3d_coordinates_for_first_frag.propagate = False
        self.error_getting_3d_coordinates_for_second_frag = logging.getLogger('log_eight')
        self.error_getting_3d_coordinates_for_second_frag.propagate = False
        self.error_standardizing_smiles_for_parent = logging.getLogger('log_nine')
        self.error_standardizing_smiles_for_parent.propagate = False
        self.error_standardizing_smiles_for_first_frag = logging.getLogger('log_ten')
        self.error_standardizing_smiles_for_first_frag.propagate = False
        self.error_standardizing_smiles_for_second_frag = logging.getLogger('log_eleven')
        self.error_standardizing_smiles_for_second_frag.propagate = False
        self.finally_used = logging.getLogger('log_twelve')
        self.finally_used.propagate = False

    def read_data_from_csv(self, paired_data_csv):
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
                    backed_parent, backed_first_frag, backed_second_frag = self.read_mol(sdf_name,
                                                                                         path_pdb_sdf_files,
                                                                                         row[col_parent_smi],
                                                                                         row[col_first_frag_smi],
                                                                                         row[col_second_frag_smi],
                                                                                         row[col_first_ligand_template],
                                                                                         row[col_second_ligand_template])
                    if not backed_parent or (not backed_first_frag and not backed_second_frag):
                        continue

                    parent_smi = Chem.MolToSmiles(backed_parent)
                    first_frag_smi = Chem.MolToSmiles(backed_first_frag) if backed_first_frag else None
                    try:
                        second_frag_smi = Chem.MolToSmiles(backed_second_frag) if backed_second_frag else None
                    except:
                        backed_second_frag = None
                        second_frag_smi = None
                    act_first_frag_smi = row[col_act_first_frag_smi] if backed_first_frag else None
                    act_second_frag_smi = row[col_act_second_frag_smi] if backed_second_frag else None
                    prevalence_receptor = row[col_prevalence] if col_prevalence else 1

                    key_sdf_pdb = self.__get_key_sdf_pdb(pdb_name, sdf_name)
                    key_parent_sdf_pdb = self.__get_key_parent_sdf_pdb(pdb_name, sdf_name, parent_smi)

                    if pdb_name not in self.pdb_files:
                        self.pdb_files.append(pdb_name)
                        self.sdf_x_pdb[pdb_name] = []
                    if sdf_name not in self.sdf_x_pdb[pdb_name]:
                        self.sdf_x_pdb[pdb_name].append(sdf_name)
                        self.parent_x_sdf_x_pdb[key_sdf_pdb] = []
                    if parent_smi not in self.parent_x_sdf_x_pdb[key_sdf_pdb]:
                        self.parent_x_sdf_x_pdb[key_sdf_pdb].append(parent_smi)
                        self.backed_mol_x_parent[parent_smi] = backed_parent
                        self.frag_and_act_x_parent_x_sdf_x_pdb[key_parent_sdf_pdb] = []
                        if backed_first_frag:
                            self.frag_and_act_x_parent_x_sdf_x_pdb[key_parent_sdf_pdb].append([first_frag_smi, act_first_frag_smi, backed_first_frag, prevalence_receptor])
                        if backed_second_frag:
                            self.frag_and_act_x_parent_x_sdf_x_pdb[key_parent_sdf_pdb].append([second_frag_smi, act_second_frag_smi, backed_second_frag, prevalence_receptor])

                    self.finally_used.info("Receptor in " + pdb_name + " and Ligand in " + sdf_name + " were used")

        self.pdb_files.sort()

    def __get_key_sdf_pdb(self, pdb_name, sdf_name):
        return pdb_name + "_" + sdf_name

    def __get_key_parent_sdf_pdb(self, pdb_name, sdf_name, parent_smi):
        return pdb_name + "_" + sdf_name + "_" + parent_smi

    def read_mol(self, sdf_name, path_pdb_sdf_files, parent_smi, first_frag_smi, second_frag_smi, first_ligand_template, second_ligand_template):
        path_to_mol = path_pdb_sdf_files + os.sep + sdf_name
        if sdf_name.endswith(".pdb"):
            try:
                ref_mol = AllChem.MolFromPDBFile(path_to_mol, removeHs=False)
                template = Chem.MolFromSmiles(first_ligand_template)
                ref_mol = AllChem.AssignBondOrdersFromTemplate(template, ref_mol)
            except:
                self.fail_match_FirstSmiles_PDBLigand.info("Ligand in " + sdf_name + " and First SMILES string (" + first_ligand_template + ") did not match")
                try:
                    ref_mol = AllChem.MolFromPDBFile(path_to_mol, removeHs=False)
                    template = Chem.MolFromSmiles(second_ligand_template)
                    ref_mol = AllChem.AssignBondOrdersFromTemplate(template, ref_mol)
                except:
                    self.fail_match_SecondSmiles_PDBLigand.info("Ligand in " + sdf_name + " and Second SMILES string (" + second_ligand_template + ") did not match")
                    return None, None, None
        elif sdf_name.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(path_to_mol)
            for ref_mol in suppl:
                pass
        else:
            return None, None, None

        # it is needed to get 3D coordinates for parent to voxelize.
        r_parent = self.get_sub_mol(ref_mol, parent_smi, sdf_name, self.ligand_not_contain_parent, self.error_getting_3d_coordinates_for_parent)

        # it is needed to get 3D coordinates for fragments to compute distance to receptor for filtering purposes.
        r_first_frag_smi = self.get_sub_mol(ref_mol, first_frag_smi, sdf_name, self.ligand_not_contain_first_frag, self.error_getting_3d_coordinates_for_first_frag)
        r_second_frag_smi = self.get_sub_mol(ref_mol, second_frag_smi, sdf_name, self.ligand_not_contain_second_frag, self.error_getting_3d_coordinates_for_second_frag)

        return self.__get_backed_molecule(molecule=r_parent, logger=self.error_standardizing_smiles_for_parent), self.__get_backed_molecule(molecule=r_first_frag_smi, logger=self.error_standardizing_smiles_for_first_frag), self.__get_backed_molecule(molecule=r_second_frag_smi, logger=self.error_standardizing_smiles_for_second_frag)

    def __get_backed_molecule(self, molecule, logger):
        try:
            if molecule:
                smi = Chem.MolToSmiles(molecule, isomericSmiles=True)
                smi = self.__standardize_smiles(smiles=smi, raise_if_fails=True)
                return molecule
        except Exception as e:
            logger.info(f"CAUGHT EXCEPTION: Could not standardize SMILES: {Chem.MolToSmiles(molecule)} >> ", e)
            return None

    # From https://www.rdkit.org/docs/Cookbook.html
    def __neutralize_atoms(self, mol: Chem.Mol) -> Chem.Mol:
        """Neutralize the molecule by adding/removing hydrogens.

        Args:
            mol (Chem.Mol): RDKit molecule.

        Returns:
            Chem.Mol: Neutralized RDKit molecule.
        """
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if at_matches_list:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        return mol

    def __standardize_smiles(self, smiles: str, raise_if_fails: bool = False) -> str:
        """Standardize SMILES string.

        Args:
            smiles (str): SMILES string.
            raise_if_fails (bool): If True, will raise an Exception.

        Returns:
            str: Standardized SMILES string.
        """
        # Convert smiles to rdkit mol
        rdmol = Chem.MolFromSmiles(smiles)

        # Neutralize the molecule (charges)
        self.__neutralize_atoms(rdmol)

        rdmolops.Cleanup(rdmol)
        rdmolops.RemoveStereochemistry(rdmol)
        rdmolops.SanitizeMol(rdmol, catchErrors=True)

        # Remove hydrogens
        rdmol = Chem.RemoveHs(rdmol)

        return Chem.MolToSmiles(
            rdmol,
            isomericSmiles=False,  # No chirality
            canonical=True,  # e.g., all benzenes are written as aromatic
        )

    # mol must be RWMol object
    # based on https://github.com/wengong-jin/hgraph2graph/blob/master/hgraph/chemutils.py
    def get_sub_mol(self, mol, smi_sub_mol, sdf_name, log_for_fragments, log_for_3d_coordinates):
        patt = Chem.MolFromSmarts(smi_sub_mol)
        mol_smile = Chem.MolToSmiles(patt)
        mol_smile = mol_smile.replace("*", "[H]")
        patt_mol = Chem.MolFromSmiles(mol_smile)
        Chem.RemoveAllHs(patt_mol)
        sub_atoms = mol.GetSubstructMatch(patt_mol, useChirality=True)
        if len(sub_atoms) == 0:
            log_for_fragments.info("Ligand " + sdf_name + " has not the fragment " + Chem.MolToSmiles(patt))
            return None

        new_mol = Chem.RWMol()
        atom_map = {}
        for idx in sub_atoms:
            atom = mol.GetAtomWithIdx(idx)
            atom_map[idx] = new_mol.AddAtom(atom)

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

        try:
            new_mol = new_mol.GetMol()
            new_mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(new_mol,
                             Chem.SanitizeFlags.SANITIZE_ADJUSTHS | Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                             catchErrors=False)
            Chem.Kekulize(new_mol, clearAromaticFlags=False)
            new_mol = Chem.MolToMolBlock(new_mol)
            new_mol = Chem.MolFromMolBlock(new_mol)
            conf = new_mol.GetConformer()
            for idx in sub_atoms:
                a = mol.GetAtomWithIdx(idx)
                x, y, z = mol.GetConformer().GetAtomPosition(idx)
                conf.SetAtomPosition(atom_map[a.GetIdx()], Point3D(x, y, z))
        except Exception as e:
            log_for_3d_coordinates.info("3D coordinates of the fragment " + Chem.MolToSmiles(new_mol) + " cannot be extracted from the ligand " + sdf_name + " because " + str(e))
            return None

        # find out the connector atom
        # NOTE: according to several runs, the connector atom is always allocated in the position 0 into the recovered substructure ('new_mol' variable)
        # but this was implemented just in case the connector atom is in a position other than 0.
        # this implementation has linear complexity and it is so fast
        for idx in sub_atoms:
            a = mol.GetAtomWithIdx(idx)
            if patt.GetAtomWithIdx(atom_map[a.GetIdx()]).GetSymbol() == "*":  # this is the connector atom
                new_mol.GetAtomWithIdx(atom_map[a.GetIdx()]).SetAtomicNum(0)

        return new_mol

    def __setup_logger(self, logger_name, log_file, level=logging.INFO):

        log_setup = logging.getLogger(logger_name)
        formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        log_setup.setLevel(level)
        log_setup.addHandler(file_handler)
        # log_setup.addHandler(stream_handler)


if __name__ == "__main__":
    paired_obj = PairedPdbSdfCsvInterface()
    paired_obj.read_data_from_csv("D:\\Cesar\\0.Investigacion\\3.Experimentacion\\DeepFrag\\paired_data_original.txt,D:\\Cesar\\0.Investigacion\\3.Experimentacion\\DeepFrag\\Datasets_paired,protein_pdb,ligand_sdf,Parent,FirstFragment,SecondFragment,Endpoint_First,Endpoint_Second,Template_one,Template_two,ReceptorPrevalence")
    sys.exit()
