from typing import Union
from collagen.core.molecules.mol import BackedMol
import os
from pathlib import Path
import logging
import csv
from collagen.external.common.parent_interface import ParentInterface
from collagen.external.common.types import StructuresClass, StructuresFamily
from collagen.external.paired_csv.targets_ligands import (
    PairedCsv_ligand,
    PairedCsv_target,
)
from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
import numpy as np  # type: ignore


class PairedCsvInterface(ParentInterface):
    pdb_files = []
    sdf_x_pdb = {}
    parent_x_sdf_x_pdb = {}
    frag_and_act_x_parent_x_sdf_x_pdb = {}
    backed_mol_x_parent = {}

    error_loading_parents = None
    error_loading_first_fragments = None
    error_loading_second_fragments = None
    finally_used = None

    def __init__(
        self,
        structures: Union[str, Path],
        cache_pdbs_to_disk: bool,
        grid_width: int,
        grid_resolution: float,
        noh: bool,
        discard_distant_atoms: bool,
        use_prevalence: bool,
    ):
        # Make sure it is string, not Path
        if isinstance(structures, Path):
            structures = str(structures)

        self.use_prevalence = use_prevalence

        super().__init__(
            structures,
            structures.split(",")[1],
            cache_pdbs_to_disk,
            grid_width,
            grid_resolution,
            noh,
            discard_distant_atoms,
        )

    def _creating_logger_files(self):
        self.__setup_logger(
            "log_one", os.getcwd() + os.sep + "01_error_loading_parents.log"
        )
        self.__setup_logger(
            "log_two", os.getcwd() + os.sep + "02_error_loading_first_fragments.log"
        )
        self.__setup_logger(
            "log_three", os.getcwd() + os.sep + "03_error_loading_second_fragments.log"
        )
        self.__setup_logger("log_four", os.getcwd() + os.sep + "04_finally_used.log")

        self.error_loading_parents = logging.getLogger("log_one")
        self.error_loading_parents.propagate = False
        self.error_loading_first_fragments = logging.getLogger("log_two")
        self.error_loading_first_fragments.propagate = False
        self.error_loading_second_fragments = logging.getLogger("log_three")
        self.error_loading_second_fragments.propagate = False
        self.finally_used = logging.getLogger("log_four")
        self.finally_used.propagate = False

    def _load_targets_ligands_hierarchically(
        self,
        metadata,
        cache_pdbs_to_disk: bool,
        grid_width: int,
        grid_resolution: float,
        noh: bool,
        discard_distant_atoms: bool,
    ):
        self.__read_data_from_csv(metadata)

        classes = []
        curr_class = None
        curr_class_name = None
        curr_family = None
        curr_family_name = None
        curr_target = None
        curr_target_name = None

        for full_pdb_name in self.pdb_files:
            if (curr_target is None) or (full_pdb_name != curr_target_name):
                if curr_target is not None and curr_family is not None:
                    curr_family.targets.append(curr_target)
                curr_target_name = full_pdb_name
                curr_target = PairedCsv_target(
                    pdb_id=full_pdb_name,
                    ligands=[],
                    cache_pdbs_to_disk=cache_pdbs_to_disk,
                    grid_width=grid_width,
                    grid_resolution=grid_resolution,
                    noh=noh,
                    discard_distant_atoms=discard_distant_atoms,
                )

                for sdf_name in self.sdf_x_pdb[full_pdb_name]:
                    key_sdf_pdb = self.__get_key_sdf_pdb(full_pdb_name, sdf_name)
                    for parent_smi in self.parent_x_sdf_x_pdb[key_sdf_pdb]:
                        key_parent_sdf_pdb = self.__get_key_parent_sdf_pdb(
                            full_pdb_name, sdf_name, parent_smi
                        )
                        backed_parent = self.backed_mol_x_parent[parent_smi]
                        curr_target.ligands.append(
                            PairedCsv_ligand(
                                name=key_parent_sdf_pdb,
                                validity="valid",
                                # affinity_measure="",
                                # affinity_value="",
                                # affinity_unit="",
                                smiles=parent_smi,
                                rdmol=backed_parent.rdmol,
                                fragment_and_act=self.frag_and_act_x_parent_x_sdf_x_pdb[
                                    key_parent_sdf_pdb
                                ],
                                backed_parent=backed_parent,
                            )
                        )

            if (curr_family is None) or (full_pdb_name != curr_family_name):
                if curr_family is not None and curr_class is not None:
                    curr_class.families.append(curr_family)
                curr_family_name = full_pdb_name
                curr_family = StructuresFamily(rep_pdb_id=full_pdb_name, targets=[])

            if curr_class is None:
                curr_class_name = full_pdb_name
                curr_class = StructuresClass(ec_num=full_pdb_name, families=[])
            elif full_pdb_name != curr_class_name:
                classes.append(curr_class)
                curr_class_name = full_pdb_name
                curr_class = StructuresClass(ec_num=full_pdb_name, families=[])

        if curr_target is not None and curr_family is not None:
            curr_family.targets.append(curr_target)
        if curr_family is not None and curr_class is not None:
            curr_class.families.append(curr_family)
        if curr_class is not None:
            classes.append(curr_class)

        self.classes = classes

    def _get_structure_file_extension(self) -> Union[str, None]:
        return "pdb"

    def __read_data_from_csv(self, paired_data_csv):
        # reading input parameters
        paired_data_csv_sep = paired_data_csv.split(",")
        path_csv_file = paired_data_csv_sep[0]  # path to the csv (or tab) file
        path_pdb_sdf_files = paired_data_csv_sep[
            1
        ]  # path containing the pdb and/or sdf files
        col_pdb_name = paired_data_csv_sep[2]  # pdb file name containing the receptor
        col_sdf_name = paired_data_csv_sep[
            3
        ]  # sdf (or pdb) file name containing the ligand
        col_parent_smi = paired_data_csv_sep[4]  # SMILES string for the parent
        col_first_frag_smi = paired_data_csv_sep[
            5
        ]  # SMILES string for the first fragment
        col_second_frag_smi = paired_data_csv_sep[
            6
        ]  # SMILES string for the second fragment
        col_act_first_frag_smi = paired_data_csv_sep[
            7
        ]  # activity for the first fragment
        col_act_second_frag_smi = paired_data_csv_sep[
            8
        ]  # activity for the second fragment
        col_first_ligand_template = paired_data_csv_sep[
            9
        ]  # first SMILES string for assigning bonds to the ligand
        col_second_ligand_template = paired_data_csv_sep[
            10
        ]  # if fail the previous SMILES, second SMILES string for assigning bonds to the ligand
        col_gen_book_id = paired_data_csv_sep[11]  # Gene Book ID column

        total_entries = 0
        gene_book_id_counter = {}
        with open(path_csv_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter="\t")
            for row in reader:
                pdb_name = row[col_pdb_name]
                sdf_name = row[col_sdf_name]
                if os.path.exists(
                    path_pdb_sdf_files + os.sep + pdb_name
                ) and os.path.exists(path_pdb_sdf_files + os.sep + sdf_name):
                    (
                        backed_parent,
                        backed_first_frag,
                        backed_second_frag,
                    ) = self.read_mol(
                        sdf_name,
                        path_pdb_sdf_files,
                        row[col_parent_smi],
                        row[col_first_frag_smi],
                        row[col_second_frag_smi],
                        row[col_first_ligand_template],
                        row[col_second_ligand_template],
                    )

                    if not backed_parent:
                        continue

                    # getting the smiles for parent.
                    try:
                        parent_smi = Chem.MolToSmiles(
                            backed_parent.rdmol, isomericSmiles=True
                        )
                    except:
                        if self.error_loading_parents is not None:
                            self.error_loading_parents.info(
                                f"CAUGHT EXCEPTION: Could not standardize SMILES: {Chem.MolToSmiles(backed_parent.rdmol)}"
                            )
                        continue

                    # getting the smiles for first fragment.
                    first_frag_smi = None
                    if backed_first_frag:
                        try:
                            first_frag_smi = Chem.MolToSmiles(
                                backed_first_frag.rdmol, isomericSmiles=True
                            )
                        except:
                            if self.error_loading_first_fragments is not None:
                                self.error_loading_first_fragments.info(
                                    f"CAUGHT EXCEPTION: Could not standardize SMILES: {Chem.MolToSmiles(backed_first_frag.rdmol)}"
                                )
                            backed_first_frag = None
                    else:
                        if self.error_loading_first_fragments is not None:
                            self.error_loading_first_fragments.info(
                                "First fragment was not read"
                            )

                    # getting the smiles for second fragment.
                    second_frag_smi = None
                    if backed_second_frag:
                        try:
                            second_frag_smi = Chem.MolToSmiles(
                                backed_second_frag.rdmol, isomericSmiles=True
                            )
                        except:
                            if self.error_loading_second_fragments is not None:
                                self.error_loading_second_fragments.info(
                                    f"CAUGHT EXCEPTION: Could not standardize SMILES: {Chem.MolToSmiles(backed_second_frag.rdmol)}"
                                )
                            backed_second_frag = None
                    else:
                        if self.error_loading_first_fragments is not None:
                            self.error_loading_first_fragments.info(
                                "Second fragment was not read"
                            )

                    if not backed_first_frag and not backed_second_frag:
                        continue

                    act_first_frag_smi = (
                        row[col_act_first_frag_smi] if backed_first_frag else None
                    )
                    act_second_frag_smi = (
                        row[col_act_second_frag_smi] if backed_second_frag else None
                    )
                    gen_book_id = row[col_gen_book_id]
                    prevalence_receptor = float(0.0)

                    key_sdf_pdb = self.__get_key_sdf_pdb(pdb_name, sdf_name)
                    key_parent_sdf_pdb = self.__get_key_parent_sdf_pdb(
                        pdb_name, sdf_name, parent_smi
                    )

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
                            self.frag_and_act_x_parent_x_sdf_x_pdb[
                                key_parent_sdf_pdb
                            ].append(
                                [
                                    first_frag_smi,
                                    act_first_frag_smi,
                                    backed_first_frag,
                                    prevalence_receptor,
                                    gen_book_id,
                                ]
                            )
                        if backed_second_frag:
                            self.frag_and_act_x_parent_x_sdf_x_pdb[
                                key_parent_sdf_pdb
                            ].append(
                                [
                                    second_frag_smi,
                                    act_second_frag_smi,
                                    backed_second_frag,
                                    prevalence_receptor,
                                    gen_book_id,
                                ]
                            )

                    if self.finally_used is not None:
                        self.finally_used.info(
                            "Receptor in "
                            + pdb_name
                            + " and Ligand in "
                            + sdf_name
                            + " were used"
                        )
                    if self.use_prevalence:
                        total_entries = total_entries + 1
                        if gen_book_id not in gene_book_id_counter.keys():
                            gene_book_id_counter[gen_book_id] = 0
                        gene_book_id_counter[gen_book_id] = (
                            gene_book_id_counter[gen_book_id] + 1
                        )

        if self.use_prevalence:
            for (
                list_for_parent_sdf_pdb
            ) in self.frag_and_act_x_parent_x_sdf_x_pdb.values():
                for entry in list_for_parent_sdf_pdb:
                    entry[3] = float(gene_book_id_counter[entry[4]] / total_entries)
                    print(
                        str(gene_book_id_counter[entry[4]])
                        + " / "
                        + str(total_entries)
                        + " = "
                        + str(entry[3])
                    )

        self.pdb_files.sort()

    def __get_key_sdf_pdb(self, pdb_name, sdf_name):
        return pdb_name + "_" + sdf_name

    def __get_key_parent_sdf_pdb(self, pdb_name, sdf_name, parent_smi):
        return pdb_name + "_" + sdf_name + "_" + parent_smi

    def __parent_smarts_to_mol(self, smi):
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
                    break

            # Now dummy atom removed, but connection marked.
            mol.UpdatePropertyCache()
            Chem.GetSymmSSSR(mol)
            return mol
        except:
            return None

    def __remove_mult_bonds_by_smi_to_smi(self, smi):
        smi = smi.upper()
        smi = smi.replace("=", "")
        smi = smi.replace("#", "")
        smi = smi.replace("BR", "Br").replace("CL", "Cl")
        return smi

    def __remove_mult_bonds(self, mol):
        # mol = Chem.MolFromSmiles(smi)
        emol = Chem.EditableMol(mol)
        for bond in mol.GetBonds():
            emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            emol.AddBond(
                bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE
            )

        mol = emol.GetMol()
        Chem.SanitizeMol(mol)
        # mol=Chem.AddHs(mol)
        return mol

    def __substruct_with_coords(self, mol, substruct_mol, atom_indices):
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

    def read_mol(
        self,
        sdf_name,
        path_pdb_sdf_files,
        parent_smi,
        first_frag_smi,
        second_frag_smi,
        first_ligand_template,
        second_ligand_template,
    ):
        path_to_mol = path_pdb_sdf_files + os.sep + sdf_name
        if sdf_name.endswith(".pdb"):

            pdb_mol = AllChem.MolFromPDBFile(path_to_mol, removeHs=False)
            if pdb_mol is None:
                # In at least one case, the pdb_mol appears to be unparsable. Must skip.
                return None, None, None

            # Get parent mol too.
            first_parent = parent_smi

            # Note that it's important to use MolFromSmarts here, not MolFromSmiles
            parent_mol = self.__parent_smarts_to_mol(first_parent)

            try:
                # Check if substructure match
                atom_indices = pdb_mol.GetSubstructMatch(
                    parent_mol, useChirality=False, useQueryQueryMatches=False
                )
                atom_indices = None if len(atom_indices) == 0 else atom_indices
            except:
                atom_indices = None

            if atom_indices is None:
                # Previous attempt failed. Try converting everything into single bonds. For parent molecule,
                # do on level of smiles to avoid errors.
                parent_smi = self.__remove_mult_bonds_by_smi_to_smi(parent_smi)
                parent_mol = self.__parent_smarts_to_mol(parent_smi)

                # Try converting everything into single bonds in ligand.
                pdb_mol = self.__remove_mult_bonds(pdb_mol)

                # Note: Not necessary to remove chirality given useChirality=False flag below.
                try:
                    atom_indices = pdb_mol.GetSubstructMatch(
                        parent_mol, useChirality=False, useQueryQueryMatches=False
                    )
                    atom_indices = None if len(atom_indices) == 0 else atom_indices
                except:
                    atom_indices = None

            if (
                atom_indices is not None
                and parent_mol is not None
                and len(atom_indices) == parent_mol.GetNumAtoms()
            ):

                # Success in finding substructure. Make new mol of just substructure.
                new_mol = self.__substruct_with_coords(
                    pdb_mol, parent_mol, atom_indices
                )

                # Get the connection point and add it to the data row
                atom_idx = -1  # So never unbound
                for atom in new_mol.GetAtoms():
                    if (
                        atom.HasProp("was_dummy_connected")
                        and atom.GetProp("was_dummy_connected") == "yes"
                    ):
                        atom_idx = atom.GetIdx()
                        break

                # atom_idx must not be -1
                assert atom_idx != -1, "Atom index must not be -1"

                conf = new_mol.GetConformer()
                connect_coord = conf.GetAtomPosition(atom_idx)
                connect_coord = np.array(
                    [connect_coord.x, connect_coord.y, connect_coord.z]
                )

                backed_parent = BackedMol(rdmol=new_mol)

                # first_frag_smi = self.__remove_mult_bonds_by_smi_to_smi(first_frag_smi)
                # first_frag_smi = self.__parent_smarts_to_mol(first_frag_smi)
                backed_frag1 = (
                    BackedMol(
                        rdmol=Chem.MolFromSmiles(first_frag_smi.replace("[R*]", "*")),
                        warn_no_confs=False,
                        coord_connector_atom=connect_coord,
                    )
                    if first_frag_smi
                    else None
                )

                # second_frag_smi = self.__remove_mult_bonds_by_smi_to_smi(second_frag_smi)
                # second_frag_smi = self.__parent_smarts_to_mol(second_frag_smi)
                backed_frag2 = (
                    BackedMol(
                        rdmol=Chem.MolFromSmiles(second_frag_smi.replace("[R*]", "*")),
                        warn_no_confs=False,
                        coord_connector_atom=connect_coord,
                    )
                    if second_frag_smi
                    else None
                )

                return backed_parent, backed_frag1, backed_frag2

            else:
                if self.error_loading_parents is not None:
                    self.error_loading_parents.info(
                        "Ligand " + sdf_name + " has not parent structure " + parent_smi
                    )

        return None, None, None

    def __setup_logger(self, logger_name, log_file, level=logging.INFO):

        log_setup = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            "%(levelname)s: %(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
        )
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        log_setup.setLevel(level)
        log_setup.addHandler(file_handler)
        # log_setup.addHandler(stream_handler)
