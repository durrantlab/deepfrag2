from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union

from collagen.external.common.parent_targets_ligands import Parent_target
from collagen.external.common.types import StructuresClass


class ParentInterface(ABC):
    @abstractmethod
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
        self.classes: List[StructuresClass] = []
        self._all_targets: List["str"] = []
        self._lookup: Dict["str", "Parent_target"] = {}
        self._init_lookup()
        self._resolve_target_paths(structures_path)

    @abstractmethod
    def _load_targets_ligands_hierarchically(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get_structure_file_extension(self) -> Union[str, None]:
        pass

    def _init_lookup(self):
        """Protein/Ligand data is divided into clasess of proteins. These are
        dividied into families, which contain the individual targets. This
        iterates through this hierarchy and just maps the pdb id to the target.
        """
        for c in self.classes:
            for f in c.families:
                for t in f.targets:
                    if t is None:
                        continue
                    self._lookup[t.pdb_id.lower()] = t

        self._all_targets = list(self._lookup)

    def _resolve_target_paths(self, path: Union[str, Path]):
        """Resolve the paths to the PDBs for each target in the
        class=>family=>target hierarchy.

        Args:
            path (Union[str, Path]): Path to the directory containing the
                PDBs.
        """
        path = Path(path)

        # Map the pdb to the file on disk.
        files = {}
        if self._get_structure_file_extension():
            for fam in path.glob(f"./**/*.{self._get_structure_file_extension()}"):
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
                    if targ is None:
                        continue

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
                        print(
                            f"No structures for {k}. Is your copy of BindingMOAD complete?"
                        )

                        # Mark this target in familes for deletion
                        fam.targets[targ_idx] = None

                # Remove any Nones from the target list
                fam.targets = [t for t in fam.targets if t is not None]

    @property
    def targets(self) -> List[str]:
        """Get a list of all targets (PDB IDs) in BindingMOAD.

        Returns:
            List[str]: A list of all targets (PDB IDs) in BindingMOAD.
        """
        return self._all_targets

    def __getitem__(self, pdb_id: str) -> "Parent_target":
        """
        Fetch a specific target by PDB ID.

        Args:
            key (str): A PDB ID (case-insensitive).

        Returns:
            Parent_target: a Parent_target object if found.
        """
        assert type(pdb_id) is str, f"PDB ID must be a str (got {type(pdb_id)})"
        k = pdb_id.lower()
        assert k in self._lookup, f'Target "{k}" not found.'
        return self._lookup[k]
