from dataclasses import field
from pathlib import Path
from typing import Dict, List, Union
from .types import MOAD_family, MOAD_class, MOAD_ligand, MOAD_target, PdbSdfDir_target, PdbSdfDir_ligand
import glob
import os
from rdkit import Chem
import linecache


class MOADInterface(object):
    """
    Base class for interacting with Binding MOAD data. Initialize by passing the
    path to "every.csv" and the path to a folder containing structure files (can
    be nested).

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
        # BindingMOAD is divided into clasess of proteins. These are dividied
        # into families, which contain the individual targets. This iterates
        # through this hierarchy and just maps the pdb id to the target.

        for c in self.classes:
            for f in c.families:
                for t in f.targets:
                    self._lookup[t.pdb_id.lower()] = t

        self._all_targets = [k for k in self._lookup]

    @property
    def targets(self) -> List["str"]:
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
        every_csv_path,
        cache_pdbs_to_disk: bool,
        grid_width: int,
        grid_resolution: float,
        noh: bool,
        discard_distant_atoms: bool,
    ):
        # Note that the output of this function gets put in self.classes.

        # BindingMOAD data is loaded into protein classes, which contain
        # families, which contain the individual targets, which are associated
        # with ligands. This function sets up a heirarchical data structure that
        # preserves these relationships. The structure is comprised of nested
        # MOAD_class, MOAD_family, MOAD_target, and MOAD_ligand dataclasses.

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
        path = Path(path)

        # Map the pdb to the file on disk.
        files = {}
        for fam in path.glob(f"./**/*.{self._extension_for_resolve_paths()}"):
            if str(fam).endswith(".pkl"):
                continue
            pdbid = fam.stem  # Removes extension (.bio?)
            print("HHH", pdbid)
            if pdbid not in files:
                files[pdbid] = []
            files[pdbid].append(fam)

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
    def __init__(
            self,
            structures: Union[str, Path],
            cache_pdbs_to_disk: bool,
            grid_width: int,
            grid_resolution: float,
            noh: bool,
            discard_distant_atoms: bool,
    ):
        super().__init__(structures, structures, cache_pdbs_to_disk, grid_width, grid_resolution, noh, discard_distant_atoms)

    def _load_classes_families_targets_ligands(
        self,
        every_csv_path,
        cache_pdbs_to_disk: bool,
        grid_width: int,
        grid_resolution: float,
        noh: bool,
        discard_distant_atoms: bool,
    ):

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
                sdf_name = parts[0] + "_lig_" + parts[2]
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
