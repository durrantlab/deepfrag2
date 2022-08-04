from dataclasses import field
from pathlib import Path
from typing import Dict, List, Union
from .types import MOAD_family, MOAD_class, MOAD_ligand, MOAD_target

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
        structures: Union[str, Path],
        cache_pdbs_to_disk: bool,
        grid_width: int,
        grid_resolution: float,
        noh: bool,
        discard_distant_atoms: bool,
    ):
        self.classes = MOADInterface._load_classes_families_targets_ligands(
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
        self._resolve_paths(structures)

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

    @staticmethod
    def _load_classes_families_targets_ligands(
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

        return classes

    def _resolve_paths(self, path: Union[str, Path]):
        path = Path(path)

        # Map the pdb to the file on disk.
        files = {}
        for fam in path.glob("./**/*.bio*"):
            if str(fam).endswith(".pkl"):
                continue
            pdbid = fam.stem  # Removes extension (.bio?)
            if pdbid not in files:
                files[pdbid] = []
            files[pdbid].append(fam)

        # Associate the filename with each target in the class=>family=>target
        # hierarchy.
        for cls in self.classes:
            for fam in cls.families:
                for targ in fam.targets:
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
                        print(f"No structures for {k}")
