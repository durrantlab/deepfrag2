"""Interface for Binding MOAD data."""

from dataclasses import field
from pathlib import Path
from typing import Dict, List, Union
from .types import MOAD_family, MOAD_class, MOAD_ligand, MOAD_target, PdbSdfDir_target, PdbSdfDir_ligand, PairedPdbSdfCsv_ligand
import glob
import os
from rdkit import Chem
import linecache
import csv
from collagen.core.molecules.mol import BackedMol
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem, rdmolops
import logging
from collagen.core.molecules import smiles_utils


class MOADInterface(object):

    """Base class for interacting with Binding MOAD data. Initialize by passing
    the path to "every.csv" and the path to a folder containing structure files
    (can be nested).

    NOTE: This just interfaces with the BindingMOAD on disk. It doesn't
    fragment those ligands (see fragment_dataset.py). It doesn't calculate the
    properties of the ligands/fragments or filter them (see cache_filter.py).

    Args:
        metadata: Path to the metadata "every.csv" file.
        structures: Path to a folder container structure files.
    """

    classes: List["MOAD_class"]
    _all_targets: List["str"] = field(default_factory=list)

    # Maps PDB ID to target. No classes or families (BindingMOAD heirarchy)
    _lookup: Dict["str", "MOAD_target"] = field(default_factory=dict)

    def __init__(
        self,
        metadata: Union[str, Path],
        structures_path: Union[str, Path],
        cache_pdbs_to_disk: bool,
        grid_width: int,
        grid_resolution: float,
        noh: bool,
        discard_distant_atoms: bool,
    ):
        """Initialize a MOADInterface object.
        
        Args:
            metadata (Union[str, Path]): Path to the metadata "every.csv" file.
            structures_path (Union[str, Path]): Path to a folder container structure files.
            cache_pdbs_to_disk (bool): Whether to cache PDBs to disk.
            grid_width (int): Grid width.
            grid_resolution (float): Grid resolution.
            noh (bool): Whether to remove hydrogens.
            discard_distant_atoms (bool): Whether to discard distant atoms.
        """
        self._creating_logger_files()
        self._load_classes_families_targets_ligands(
            metadata,
            cache_pdbs_to_disk,
            grid_width,
            grid_resolution,
            noh,
            discard_distant_atoms,
        )
        self._lookup = {}
        self._all_targets = []

        self._init_lookup()
        self._resolve_paths(structures_path)

    def _init_lookup(self):
        """BindingMOAD is divided into clasess of proteins. These are dividied
        into families, which contain the individual targets. This iterates
        through this hierarchy and just maps the pdb id to the target.
        """
        for c in self.classes:
            for f in c.families:
                for t in f.targets:
                    self._lookup[t.pdb_id.lower()] = t

        self._all_targets = list(self._lookup)

    @property
    def targets(self) -> List[str]:
        """Get a list of all targets (PDB IDs) in BindingMOAD.
        
        Returns:
            List[str]: A list of all targets (PDB IDs) in BindingMOAD.
        """
        return self._all_targets

    def __getitem__(self, pdb_id: str) -> "MOAD_target":
        """
        Fetch a specific target by PDB ID.

        Args:
            key (str): A PDB ID (case-insensitive).

        Returns:
            MOAD_target: a MOAD_target object if found.
        """
        assert type(pdb_id) is str, f"PDB ID must be a str (got {type(pdb_id)})"
        k = pdb_id.lower()
        assert k in self._lookup, f'Target "{k}" not found.'
        return self._lookup[k]

    def _creating_logger_files(self):
        pass

    def _load_classes_families_targets_ligands(
        self,
        every_csv_path: Union[str, Path],
        cache_pdbs_to_disk: bool,
        grid_width: int,
        grid_resolution: float,
        noh: bool,
        discard_distant_atoms: bool,
    ):
        """BindingMOAD data is loaded into protein classes, which contain
        families, which contain the individual targets, which are associated
        with ligands. This function sets up a heirarchical data structure that
        preserves these relationships. The structure is comprised of nested
        MOAD_class, MOAD_family, MOAD_target, and MOAD_ligand dataclasses.
        
        Args:
            every_csv_path (Union[str, Path]): Path to the metadata "every.csv"
                file.
            cache_pdbs_to_disk (bool): Whether to cache PDBs to disk.
            grid_width (int): Grid width.
            grid_resolution (float): Grid resolution.
            noh (bool): Whether to remove hydrogens.
            discard_distant_atoms (bool): Whether to discard distant atoms.
        """
        # Note that the output of this function gets put in self.classes.

        with open(every_csv_path, "r") as f:
            dat = f.read().strip().split("\n")

        classes = []
        curr_class = None
        curr_family = None
        curr_target = None

        for line in dat:
            parts = line.split(",")

            if parts[0] != "":  # 1: Protein Class
                if curr_class is not None:
                    classes.append(curr_class)
                curr_class = MOAD_class(ec_num=parts[0], families=[])
            elif parts[1] != "":  # 2: Protein Family
                if curr_target is not None:
                    curr_family.targets.append(curr_target)
                if curr_family is not None:
                    curr_class.families.append(curr_family)
                curr_family = MOAD_family(rep_pdb_id=parts[2], targets=[])
                curr_target = MOAD_target(
                    pdb_id=parts[2],
                    ligands=[],
                    cache_pdbs_to_disk=cache_pdbs_to_disk,
                    grid_width=grid_width,
                    grid_resolution=grid_resolution,
                    noh=noh,
                    discard_distant_atoms=discard_distant_atoms,
                )
            elif parts[2] != "":  # 3: Protein target
                if curr_target is not None:
                    curr_family.targets.append(curr_target)

                # if "-" in parts[2]:
                    # print(parts[2], "+++++")
                # logit(f"Loading {parts[2]}", "~/work_dir/make_moad_target.txt")

                curr_target = MOAD_target(
                    pdb_id=parts[2],
                    ligands=[],
                    cache_pdbs_to_disk=cache_pdbs_to_disk,
                    grid_width=grid_width,
                    grid_resolution=grid_resolution,
                    noh=noh,
                    discard_distant_atoms=discard_distant_atoms,
                )
            elif parts[3] != "":  # 4: Ligand
                curr_target.ligands.append(
                    MOAD_ligand(
                        name=parts[3],
                        validity=parts[4],
                        affinity_measure=parts[5],
                        affinity_value=parts[7],
                        affinity_unit=parts[8],
                        smiles=parts[9],
                    )
                )

        if curr_target is not None:
            curr_family.targets.append(curr_target)
        if curr_family is not None:
            curr_class.families.append(curr_family)
        if curr_class is not None:
            classes.append(curr_class)

        self.classes = classes

    def _resolve_paths(self, path: Union[str, Path]):
        """Resolve the paths to the PDBs for each target in the
        class=>family=>target hierarchy.

        Args:
            path (Union[str, Path]): Path to the directory containing the
                PDBs.
        """
        path = Path(path)

        # Map the pdb to the file on disk.
        files = {}
        if self._extension_for_resolve_paths():
            for fam in path.glob(f"./**/*.{self._extension_for_resolve_paths()}"):
                if str(fam).endswith(".pkl"):
                    continue
                pdbid = fam.stem  # Removes extension (.bio?)

                # if pdbid.lower() != pdbid:
                #     print(f"Warning: {pdbid} has upper-case letters ({fam}). Use lower case for all filenames.")

                if pdbid not in files:
                    files[pdbid] = []
                files[pdbid].append(fam)
        else:
            files["Non"] = ["Non"]

        # Associate the filename with each target in the class=>family=>target
        # hierarchy.
        for cls in self.classes:
            for fam in cls.families:
                for targ_idx, targ in enumerate(fam.targets):
                    # Assume lower case
                    k = targ.pdb_id.lower()

                    # Added this to accomodate filenames that are not all lower
                    # case. For custom MOAD-like data.
                    if k not in files:
                        k = targ.pdb_id
                    if k not in files:
                        k = k.split(".pdb")[0]
                    if k not in files:
                        k = k.split(".PDB")[0]

                    if k in files:
                        targ.files = sorted(files[k])
                    else:
                        # No structures for this pdb id!
                        print(f"No structures for {k}. Is your copy of BindingMOAD complete?")

                        # Mark this target in familes for deletion
                        fam.targets[targ_idx] = None

                # Remove any Nones from the target list
                fam.targets = [t for t in fam.targets if t is not None]

    def _extension_for_resolve_paths(self):
        return "bio*"


class PdbSdfDirInterface(MOADInterface):

    """Interface for data stored in a directory of PDBs and SDFs."""

    def __init__(
            self,
            structures_dir: Union[str, Path],
            cache_pdbs_to_disk: bool,
            grid_width: int,
            grid_resolution: float,
            noh: bool,
            discard_distant_atoms: bool,
    ):
        """Interface for data stored in a directory of PDBs and SDFs.

        Args:
            structures_dir (Union[str, Path]): Path to the directory containing the
                PDBs and SDFs.
            cache_pdbs_to_disk (bool): Whether to cache the PDBs to disk.
            grid_width (int): Width of the grid.
            grid_resolution (float): Resolution of the grid.
            noh (bool): Whether to remove hydrogens.
            discard_distant_atoms (bool): Whether to discard distant atoms.
        """
        super().__init__(structures_dir, structures_dir, cache_pdbs_to_disk, grid_width, grid_resolution, noh, discard_distant_atoms)

    def _load_classes_families_targets_ligands(
        self,
        every_csv_path: Union[str, Path],
        cache_pdbs_to_disk: bool,
        grid_width: int,
        grid_resolution: float,
        noh: bool,
        discard_distant_atoms: bool,
    ):
        """Load the classes, families, targets, and ligands from the
        CSV file.

        Args:
            every_csv_path (Union[str, Path]): Path to the CSV file. TODO: Is
                this right? Confusing name?
            cache_pdbs_to_disk (bool): Whether to cache the PDBs to disk.
            grid_width (int): Width of the grid.
            grid_resolution (float): Resolution of the grid.
            noh (bool): Whether to remove hydrogens.
            discard_distant_atoms (bool): Whether to discard distant atoms.
        """
        classes = []
        curr_class = None
        curr_class_name = None
        curr_family = None
        curr_family_name = None
        curr_target = None
        curr_target_name = None

        pdb_files = glob. glob(every_csv_path + os.sep + "*.pdb", recursive=True)
        pdb_files.sort()
        for line in pdb_files:
            parts = line.split(os.sep)
            full_pdb_name = parts[len(parts) - 1].split(".")[0]
            parts = full_pdb_name.split("_")

            if (curr_target is None) or (full_pdb_name != curr_target_name):
                if curr_target is not None:
                    curr_family.targets.append(curr_target)
                curr_target_name = full_pdb_name
                curr_target = PdbSdfDir_target(
                    pdb_id=full_pdb_name,
                    ligands=[],
                    cache_pdbs_to_disk=cache_pdbs_to_disk,
                    grid_width=grid_width,
                    grid_resolution=grid_resolution,
                    noh=noh,
                    discard_distant_atoms=discard_distant_atoms,
                )
                sdf_name = f"{parts[0]}_lig_{parts[2]}"
                sdf_reader = Chem.SDMolSupplier(every_csv_path + os.sep + sdf_name + ".sdf")
                for ligand_ in sdf_reader:  # it is expected only one iteration because each SDF file must have a single molecule
                    if ligand_ is not None:
                        curr_target.ligands.append(
                            PdbSdfDir_ligand(
                                name=linecache.getline(every_csv_path + os.sep + sdf_name + ".sdf", 1).rstrip(),
                                validity="valid",
                                affinity_measure="",
                                affinity_value="",
                                affinity_unit="",
                                smiles=Chem.MolToSmiles(ligand_),
                                rdmol=ligand_,
                            )
                        )

            if (curr_family is None) or (parts[0] != curr_family_name):
                if curr_family is not None:
                    curr_class.families.append(curr_family)
                curr_family_name = parts[0]
                curr_family = MOAD_family(rep_pdb_id=parts[0], targets=[])

            if curr_class is None:
                curr_class_name = parts[0]
                curr_class = MOAD_class(ec_num=parts[0], families=[])
            elif parts[0] != curr_class_name:
                classes.append(curr_class)
                curr_class_name = parts[0]
                curr_class = MOAD_class(ec_num=parts[0], families=[])

        if curr_target is not None:
            curr_family.targets.append(curr_target)
        if curr_family is not None:
            curr_class.families.append(curr_family)
        if curr_class is not None:
            classes.append(curr_class)

        self.classes = classes

    def _extension_for_resolve_paths(self):
        return "pdb"


class PairedPdbSdfCsvInterface(MOADInterface):
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
            structures: Union[str, Path],
            cache_pdbs_to_disk: bool,
            grid_width: int,
            grid_resolution: float,
            noh: bool,
            discard_distant_atoms: bool,
    ):
        super().__init__(structures, structures.split(",")[1], cache_pdbs_to_disk, grid_width, grid_resolution, noh, discard_distant_atoms)

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

    def _load_classes_families_targets_ligands(
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
                if curr_target is not None:
                    curr_family.targets.append(curr_target)
                curr_target_name = full_pdb_name
                curr_target = PdbSdfDir_target(
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
                        key_parent_sdf_pdb = self.__get_key_parent_sdf_pdb(full_pdb_name, sdf_name, parent_smi)
                        backed_parent = self.backed_mol_x_parent[parent_smi]
                        curr_target.ligands.append(
                            PairedPdbSdfCsv_ligand(
                                name=key_parent_sdf_pdb,
                                validity="valid",
                                affinity_measure="",
                                affinity_value="",
                                affinity_unit="",
                                smiles=parent_smi,
                                rdmol=backed_parent.rdmol,
                                fragment_and_act=self.frag_and_act_x_parent_x_sdf_x_pdb[key_parent_sdf_pdb],
                                backed_parent=backed_parent,
                            )
                        )

            if (curr_family is None) or (full_pdb_name != curr_family_name):
                if curr_family is not None:
                    curr_class.families.append(curr_family)
                curr_family_name = full_pdb_name
                curr_family = MOAD_family(rep_pdb_id=full_pdb_name, targets=[])

            if curr_class is None:
                curr_class_name = full_pdb_name
                curr_class = MOAD_class(ec_num=full_pdb_name, families=[])
            elif full_pdb_name != curr_class_name:
                classes.append(curr_class)
                curr_class_name = full_pdb_name
                curr_class = MOAD_class(ec_num=full_pdb_name, families=[])

        if curr_target is not None:
            curr_family.targets.append(curr_target)
        if curr_family is not None:
            curr_class.families.append(curr_family)
        if curr_class is not None:
            classes.append(curr_class)

        self.classes = classes

    def _extension_for_resolve_paths(self):
        return "pdb"

    def __read_data_from_csv(self, paired_data_csv):
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
                    backed_parent, backed_first_frag, backed_second_frag = self.read_mol(sdf_name,
                                                                                         path_pdb_sdf_files,
                                                                                         row[col_parent_smi],
                                                                                         row[col_first_frag_smi],
                                                                                         row[col_second_frag_smi],
                                                                                         row[col_first_ligand_template],
                                                                                         row[col_second_ligand_template])

                    if not backed_parent:
                        continue

                    # getting the smiles for parent.
                    try:
                        parent_smi = Chem.MolToSmiles(backed_parent.rdmol, isomericSmiles=True)
                    except:
                        self.error_standardizing_smiles_for_parent.info(f"CAUGHT EXCEPTION: Could not standardize SMILES: {Chem.MolToSmiles(backed_parent.rdmol)}")
                        continue

                    # getting the smiles for first fragment.
                    if backed_first_frag:
                        try:
                            first_frag_smi = Chem.MolToSmiles(backed_first_frag.rdmol, isomericSmiles=True)
                        except:
                            self.error_standardizing_smiles_for_first_frag.info(f"CAUGHT EXCEPTION: Could not standardize SMILES: {Chem.MolToSmiles(backed_first_frag.rdmol)}")
                            backed_first_frag = None

                    # getting the smiles for second fragment.
                    if backed_second_frag:
                        try:
                            second_frag_smi = Chem.MolToSmiles(backed_second_frag.rdmol, isomericSmiles=True)
                        except:
                            self.error_standardizing_smiles_for_second_frag.info(f"CAUGHT EXCEPTION: Could not standardize SMILES: {Chem.MolToSmiles(backed_second_frag.rdmol)}")
                            backed_second_frag = None

                    if not backed_first_frag and not backed_second_frag:
                        continue

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

            # Now dummy atom removed, but connection marked.
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
            emol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)

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

    def read_mol(self, sdf_name, path_pdb_sdf_files, parent_smi, first_frag_smi, second_frag_smi, first_ligand_template, second_ligand_template):
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
                atom_indices = pdb_mol.GetSubstructMatch(parent_mol, useChirality=False, useQueryQueryMatches=False)
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
                    atom_indices = pdb_mol.GetSubstructMatch(parent_mol, useChirality=False, useQueryQueryMatches=False)
                    atom_indices = None if len(atom_indices) == 0 else atom_indices
                except:
                    atom_indices = None

            if atom_indices is not None and len(atom_indices) == parent_mol.GetNumAtoms():

                # Success in finding substructure. Make new mol of just substructure.
                new_mol = self.__substruct_with_coords(pdb_mol, parent_mol, atom_indices)

                # Get the connection point and add it to the data row
                for atom in new_mol.GetAtoms():
                    if atom.HasProp("was_dummy_connected") and atom.GetProp("was_dummy_connected") == "yes":
                        atom_idx = atom.GetIdx()
                        break

                conf = new_mol.GetConformer()
                connect_coord = conf.GetAtomPosition(atom_idx)
                connect_coord = [connect_coord.x, connect_coord.y, connect_coord.z]

                backed_parent = BackedMol(rdmol=new_mol, coord_connector_atom=connect_coord)

                first_frag_smi = self.__remove_mult_bonds_by_smi_to_smi(first_frag_smi)
                first_frag_smi = self.__parent_smarts_to_mol(first_frag_smi)
                backed_frag1 = BackedMol(rdmol=first_frag_smi, warn_no_confs=False) if first_frag_smi else None

                second_frag_smi = self.__remove_mult_bonds_by_smi_to_smi(second_frag_smi)
                second_frag_smi = self.__parent_smarts_to_mol(second_frag_smi)
                backed_frag2 = BackedMol(rdmol=second_frag_smi, warn_no_confs=False) if second_frag_smi else None

                return backed_parent, backed_frag1, backed_frag2

            else:
                self.ligand_not_contain_parent.info("Ligand " + sdf_name + " has not parent structure " + parent_smi)

        else:
            return None, None, None

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


class SdfDirInterface(MOADInterface):

    """Interface for data stored in a directory of SDFs."""

    def __init__(
            self,
            structures_dir: Union[str, Path],
            cache_pdbs_to_disk: bool,
            grid_width: int,
            grid_resolution: float,
            noh: bool,
            discard_distant_atoms: bool,
    ):
        """Initialize the interface.
        
        Args:
            structures_dir (Union[str, Path]): Path to the directory containing the
                SDFs.
            cache_pdbs_to_disk (bool): Whether to cache the PDBs to disk.
            grid_width (int): Width of the grid.
            grid_resolution (float): Resolution of the grid.
            noh (bool): Whether to remove hydrogens.
            discard_distant_atoms (bool): Whether to discard distant atoms.
        """
        super().__init__(structures_dir, structures_dir, cache_pdbs_to_disk, grid_width, grid_resolution, noh, discard_distant_atoms)

    def _load_classes_families_targets_ligands(
        self,
        every_csv_path: Union[str, Path],  # TODO: Meaning in this context?
        cache_pdbs_to_disk: bool,
        grid_width: int,
        grid_resolution: float,
        noh: bool,
        discard_distant_atoms: bool,
    ):
        """Load the classes, families, targets, and ligands from the
        CSV file.

        Args:
            every_csv_path (Union[str, Path]): Path to the CSV file. TODO: Is
                this right? Confusing name?
            cache_pdbs_to_disk (bool): Whether to cache the PDBs to disk.
            grid_width (int): Width of the grid.
            grid_resolution (float): Resolution of the grid.
            noh (bool): Whether to remove hydrogens.
            discard_distant_atoms (bool): Whether to discard distant atoms.
        """
        # TODO: Concerned things like noh and dicard_distant_atoms not being
        # used. Need to investigate and understand this function better.

        classes = []
        curr_target = PdbSdfDir_target(
            pdb_id="Non",
            ligands=[],
            cache_pdbs_to_disk=None,
            grid_width=None,
            grid_resolution=None,
            noh=None,
            discard_distant_atoms=None,
        )

        sdf_files = glob. glob(every_csv_path + os.sep + "*.sdf", recursive=True)
        sdf_files.sort()
        for line in sdf_files:
            parts = line.split(os.sep)
            sdf_name = parts[len(parts) - 1].split(".")[0]

            sdf_reader = Chem.SDMolSupplier(every_csv_path + os.sep + sdf_name + ".sdf")
            for ligand_ in sdf_reader:  # it is expected only one iteration because each SDF file must have a single molecule
                if ligand_ is not None:
                    curr_target.ligands.append(
                        PdbSdfDir_ligand(
                            name=linecache.getline(every_csv_path + os.sep + sdf_name + ".sdf", 1).rstrip(),
                            validity="valid",
                            affinity_measure="",
                            affinity_value="",
                            affinity_unit="",
                            smiles=Chem.MolToSmiles(ligand_),
                            rdmol=ligand_,
                        )
                    )

        curr_family = MOAD_family(rep_pdb_id="Non", targets=[])
        curr_family.targets.append(curr_target)
        curr_class = MOAD_class(ec_num="Non", families=[])
        curr_class.families.append(curr_family)
        classes.append(curr_class)

        self.classes = classes

    def _extension_for_resolve_paths(self):
        return None
