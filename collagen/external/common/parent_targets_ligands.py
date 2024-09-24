from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import StringIO
import os
from pathlib import Path
import pickle
import sys
from typing import Any, List, OrderedDict, Tuple, Union
import prody  # type: ignore
from collagen.core.molecules.mol import BackedMol, Mol


@dataclass
class Parent_target(ABC):

    """Base class for a target."""

    pdb_id: str
    ligands: List["Parent_ligand"]
    files: List[Path] = field(default_factory=list)
    cache_pdbs_to_disk: bool = False
    grid_width: int = 24  # Number of grid points in each dimension.
    grid_resolution: float = (
        0.75  # Distance between neighboring grid points in angstroms.
    )
    noh: bool = False  # If true, discards hydrogen atoms
    discard_distant_atoms: bool = False
    # DeepFrag requires 3.062500, but a few extra angstroms won't hurt. Note
    # that this is effectively hard coded because never specified elsewhere.
    # Should be from max((atom_radii[i] * atom_scale)**2)
    grid_padding: float = 6.0

    recent_pickle_contents = OrderedDict()

    def __len__(self) -> int:
        """
        Return the number of on-disk structures.

        Returns:
            int: The number of on-disk structures.
        """
        return len(self.files)

    def _get_pdb_from_disk_cache(self, idx: int) -> Tuple[Any, Any]:
        # Load the protein/ligand complex (PDB formatted).
        pkl_filename = str(self.files[idx])
        if self.discard_distant_atoms:
            pkl_filename += f"_{str(self.grid_width)}_{str(self.grid_resolution)}"
        if self.noh:
            pkl_filename += "_noh"
        pkl_filename += ".pkl"

        if self.cache_pdbs_to_disk and os.path.exists(pkl_filename):
            # Check if it's been loaded recently. If so, no need to reload.
            # print(self.recent_pickle_contents.keys())
            if (
                pkl_filename in self.recent_pickle_contents
            ):  # and self.recent_pickle_contents[pkl_filename] is not None:
                payload = self.recent_pickle_contents[pkl_filename]

                # If more than 100 recent pickles in memory, remove the oldest one.
                # print(len(self.recent_pickle_contents))
                while len(self.recent_pickle_contents) > 10:
                    self.recent_pickle_contents.popitem(last=False)

                return payload, pkl_filename

            # Get it from the pickle.
            try:
                # print("Open: ", pkl_filename)
                f = open(pkl_filename, "rb")
                payload = pickle.load(f)  # [receptor, ligands]
                f.close()
                self.recent_pickle_contents[pkl_filename] = payload
                # print("Close: ", pkl_filename)
                # self.memory_cache[idx] = payload
                return payload, pkl_filename
            except Exception as e:
                # If there's an error loading the pickle file, regenerate the
                # pickle file
                print(f"Error loading pkl: {str(e)}", file=sys.stderr)

        return None, pkl_filename

    def _load_pdb(self, idx: int) -> Any:
        # Returns prody molecule

        # If you get here, either you're not caching PDBs, or there is no
        # previous cached PDB you can use, or that cached file is corrupted. So
        # loading from original PDB file (slower).
        prody.LOGGER._logger.disabled = True
        if self.cache_pdbs_to_disk:
            # To improve caching, consider only lines that start with ATOM or
            # HETATM, etc. This makes files smaller and speeds pickling a bit.
            with open(self.files[idx]) as f:
                lines = [
                    l
                    # Seems being that readLines() is necessary to run on Windows
                    for l in f.readlines()
                    if l.startswith("ATOM")
                    or l.startswith("HETATM")
                    or l.startswith("MODEL")
                    or l.startswith("END")
                ]
            pdb_txt = "".join(lines)

            # Also, keep only the first model.
            pdb_txt = pdb_txt.split("ENDMDL")[0]

            # Create prody molecule.
            m = prody.parsePDBStream(StringIO(pdb_txt), model=1)
        else:
            # Not caching, so just load the file without preprocessing.
            with open(self.files[idx], "r") as f:
                m = prody.parsePDBStream(
                    f, model=1
                )  # model=1 not necessary, but just in case...
        return m

    @abstractmethod
    def _get_lig_from_prody_mol(self, lig_atoms, lig) -> Union["BackedMol", None]:
        pass

    @abstractmethod
    def _get_rec_from_prody_mol(
        self, m: Any, not_part_of_protein_sels: List[str], lig_sels: List[str]
    ) -> Union["BackedMol", None]:
        pass

    def _save_to_file_cache(self, pkl_filename: str, rec_mol: Any, ligands: Any):
        if self.cache_pdbs_to_disk:
            if os.path.exists(pkl_filename):
                print(
                    "\nFile already exists! Already saved from another thread?",
                    pkl_filename,
                )
            with open(pkl_filename, "wb") as f:
                # print("Save pickle")
                pickle.dump([rec_mol, ligands], f, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[BackedMol, List[BackedMol]]:
        pass


@dataclass
class Parent_ligand:
    """Base class to hold common information about a ligand."""

    name: str
    validity: str
    smiles: str

    @property
    def chain(self) -> str:
        """Chain ID of the ligand.

        Returns:
            str: The chain ID of the ligand.
        """
        return self.name.split(":")[1]

    @property
    def resnum(self) -> int:
        """Residue number of the ligand.

        Returns:
            int: The residue number of the ligand.
        """
        return int(self.name.split(":")[2])

    @property
    def reslength(self) -> int:
        """Length of the residue name of the ligand.

        Returns:
            int: The length of the residue name of the ligand.
        """
        return len(self.name.split(":")[0].split(" "))

    @property
    def is_valid(self) -> bool:
        """Whether the ligand is valid.

        Returns:
            bool: Whether the ligand is valid.
        """
        return self.validity == "valid"
