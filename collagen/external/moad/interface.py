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
import sys
from rdkit.Chem import AllChem


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

                    parent_smi = backed_parent.smiles(True)
                    first_frag_smi = backed_first_frag.smiles(True) if backed_first_frag else None
                    try:
                        second_frag_smi = backed_second_frag.smiles(True) if backed_second_frag else None
                    except:
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

        self.pdb_files.sort()

    def __get_key_sdf_pdb(self, pdb_name, sdf_name):
        return pdb_name + "_" + sdf_name

    def __get_key_parent_sdf_pdb(self, pdb_name, sdf_name, parent_smi):
        return pdb_name + "_" + sdf_name + "_" + parent_smi

    def read_mol(self, sdf_name, path_pdb_sdf_files, parent_smi, first_frag_smi, second_frag_smi, first_ligand_template, second_ligand_template):
        backed_parent = None
        backed_first_frag = None
        backed_second_frag = None
        path_to_mol = path_pdb_sdf_files + os.sep + sdf_name

        if sdf_name.endswith(".pdb"):
            try:
                ref_mol = AllChem.MolFromPDBFile(path_to_mol, removeHs=False)
                template = Chem.MolFromSmiles(first_ligand_template)
                ref_mol = AllChem.AssignBondOrdersFromTemplate(template, ref_mol)
            except:
                try:
                    ref_mol = AllChem.MolFromPDBFile(path_to_mol, removeHs=False)
                    template = Chem.MolFromSmiles(second_ligand_template)
                    ref_mol = AllChem.AssignBondOrdersFromTemplate(template, ref_mol)
                except:
                    # print("Molecule " + sdf_name + " was rejected because of wrong SMILES used as template to assign bonds\n")
                    return None, None, None
        elif sdf_name.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(path_pdb_sdf_files + os.sep + sdf_name)
            for ref_mol in suppl:
                pass
        else:
            return None, None, None

        r_parent = self.get_sub_mol(ref_mol, parent_smi)
        r_first_frag_smi = self.get_sub_mol(ref_mol, first_frag_smi)
        r_second_frag_smi = self.get_sub_mol(ref_mol, second_frag_smi)

        if r_parent:
            backed_parent = BackedMol(rdmol=r_parent)
        if r_first_frag_smi:
            backed_first_frag = BackedMol(rdmol=r_first_frag_smi)
        if r_second_frag_smi:
            backed_second_frag = BackedMol(rdmol=r_second_frag_smi)

        return backed_parent, backed_first_frag, backed_second_frag

    # mol must be RWMol object
    # based on https://github.com/wengong-jin/hgraph2graph/blob/master/hgraph/chemutils.py
    def get_sub_mol(self, mol, smi_sub_mol, debug=False):
        patt = Chem.MolFromSmarts(smi_sub_mol)
        sub_atoms = mol.GetSubstructMatch(patt, useChirality=True)
        if len(sub_atoms) == 0:
            # print("Molecule " + Chem.MolToSmiles(mol) + " has not the fragment " + Chem.MolToSmiles(patt), file=sys.stderr)
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
            # print("Molecule " + Chem.MolToSmiles(mol) + " and fragment " + Chem.MolToSmiles(new_mol) + " " + str(e), file=sys.stderr)
            return None

        if debug:
            print(Chem.MolToSmiles(new_mol))
            print(Chem.MolToSmiles(patt))

        # find out the connector atom
        # NOTE: according to several runs, the connector atom is always allocated in the position 0 into the recovered substructure ('new_mol' variable)
        # but this was implemented just in case the connector atom is in a position other than 0.
        # this implementation has linear complexity and it is so fast
        for idx in sub_atoms:
            a = mol.GetAtomWithIdx(idx)
            if debug:
                print(str(atom_map[a.GetIdx()]) + " " + new_mol.GetAtomWithIdx(atom_map[a.GetIdx()]).GetSymbol() + " " + patt.GetAtomWithIdx(atom_map[a.GetIdx()]).GetSymbol())
            if patt.GetAtomWithIdx(atom_map[a.GetIdx()]).GetSymbol() == "*":  # this is the connector atom
                new_mol.GetAtomWithIdx(atom_map[a.GetIdx()]).SetAtomicNum(0)

        if debug:
            for s in sub_atoms:
                print(mol.GetAtoms()[s].GetSymbol(), list(mol.GetConformer().GetAtomPosition(s)), file=sys.stderr)
            print("\n" + Chem.MolToMolBlock(new_mol), file=sys.stderr)
            print("--------------------------------------------------------------------------------------------------------", file=sys.stderr)

        return new_mol


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
