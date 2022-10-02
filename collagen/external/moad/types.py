from dataclasses import dataclass, field
from collagen.core import args as user_args
from typing import List, Tuple, Any
from pathlib import Path
import textwrap
from collagen.core.debug import logit
from collagen.core.molecules.mol import (
    TemplateGeometryMismatchException,
    UnparsableGeometryException,
    UnparsableSMILESException,
)
from rdkit import Chem
import os
import pickle
from io import StringIO
# from functools import lru_cache
import prody
from ... import Mol
from .moad_utils import fix_moad_smiles

# Simple dataclasses like MOAD_class, MOAD_family, MOAD_target, etc. Note that
# MOAD_target has some complexity to it (to load/save PDB files, including
# caching), but let's leave it here.

@dataclass
class MOAD_class(object):
    ec_num: str
    families: List["MOAD_family"]


@dataclass
class MOAD_family(object):
    rep_pdb_id: str
    targets: List["MOAD_target"]


@dataclass
class MOAD_target(object):
    pdb_id: str
    ligands: List["MOAD_ligand"]
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

    def __len__(self) -> int:
        """Returns the number of on-disk structures."""
        return len(self.files)

    def _get_pdb_from_disk_cache(self, idx: int) -> Tuple[Any]:
        # Load the protein/ligand complex (PDB formatted).

        pkl_filename = str(self.files[idx])
        if self.discard_distant_atoms:
            pkl_filename += f"_{str(self.grid_width)}_{str(self.grid_resolution)}"
        if self.noh:
            pkl_filename += "_noh"
        pkl_filename += ".pkl"

        if self.cache_pdbs_to_disk and os.path.exists(pkl_filename):
            # Get if from the pickle.
            try:
                with open(pkl_filename, "rb") as f:
                    payload = pickle.load(f)  # [receptor, ligands]
                    # self.memory_cache[idx] = payload
                    return payload, pkl_filename
            except Exception:
                # If there's an error loading the pickle file, regenerate the
                # pickle file
                # print("Corrupt pkl")
                pass

        return None, pkl_filename

    def _load_pdb(self, idx: int) -> Any:
        # Returns prody molecule

        # If you get here, either you're not caching PDBs, or there is no
        # previous cached PDB you can use, or that cached file is corrupted. So
        # loading from original PDB file (slower).

        if self.cache_pdbs_to_disk:
            # To improve caching, consider only lines that start with ATOM or
            # HETATM, etc. This makes files smaller and speeds pickling a bit.
            with open(self.files[idx]) as f:
                lines = [
                    l
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

    def _get_lig_from_prody_mol(self, lig_atoms, lig):
        try:
            lig_mol = Mol.from_prody(
                lig_atoms, fix_moad_smiles(lig.smiles), sanitize=True
            )

            if lig_mol is not None:
                lig_mol.meta["name"] = lig.name
                lig_mol.meta["moad_ligand"] = lig

            return lig_mol

        # Catch a whole bunch of errors.
        except UnparsableSMILESException as err:
            if user_args.verbose:
                msg = str(err).replace("[LIGAND]", f"{self.pdb_id}:{lig.name}")
                print(textwrap.fill(msg, subsequent_indent="  "))
            return None
        except UnparsableGeometryException as err:
            if user_args.verbose:
                msg = str(err).replace("[LIGAND]", f"{self.pdb_id}:{lig.name}")
                print(textwrap.fill(msg, subsequent_indent="  "))
            return None
        except TemplateGeometryMismatchException as err:
            if user_args.verbose:
                msg = str(err).replace("[LIGAND]", f"{self.pdb_id}:{lig.name}")
                print(textwrap.fill(msg, subsequent_indent="  "))
            return None
        except Exception as err:
            if user_args.verbose:
                msg = f"WARNING: Could not process ligand {self.pdb_id}:{lig.name}. An unknown error occured: {str(err)}"
                print(textwrap.fill(msg, subsequent_indent="  "))
            return None

    def _get_rec_from_prody_mol(
        self, m: Any, ignore_sels: List[str], lig_sels: List[str]
    ):
        if ignore_sels:
            rec_sel = "not water and not (%s)" % " or ".join(
                f"({x})" for x in ignore_sels
            )
        else:
            rec_sel = "not water"

        if self.discard_distant_atoms:
            # Only keep those portions of the receptor that are near some ligand
            # (to speed later calculations).

            all_lig_sel = "(" + ") or (".join(lig_sels) + ")"

            # Get half distance along axis
            dist = 0.5 * self.grid_width * self.grid_resolution

            # Need to account for diagnol
            dist = (3**0.5) * dist

            # Add padding
            dist = dist + self.grid_padding

            rec_sel = f"({rec_sel}) and (exwithin {str(dist)} of ({all_lig_sel}))"

        if self.noh:
            # Removing hydrogen atoms (when not needed) also speeds the
            # calculations.
            rec_sel = f"not hydrogen and ({rec_sel})"

        rec_mol = Mol.from_prody(m.select(rec_sel))
        rec_mol.meta["name"] = f"Receptor {self.pdb_id.lower()}"

        return rec_mol

    def _save_to_file_cache(self, pkl_filename: str, rec_mol: Any, ligands: Any):
        if self.cache_pdbs_to_disk:
            with open(pkl_filename, "wb") as f:
                # print("Save pickle")
                pickle.dump([rec_mol, ligands], f, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx: int) -> Tuple[Mol, Mol]:
        """
        Load the Nth structure for this target.

        Args:
            idx (int): The index of the biological assembly to load.

        Returns a (receptor, ligand) tuple of
            :class:`atlas.data.molecules.mol.Mol` objects.
        """

        # First try loading from the cache on file (.pkl).
        cached_recep_and_ligs, pkl_filename = self._get_pdb_from_disk_cache(idx)
        if cached_recep_and_ligs is not None:
            return cached_recep_and_ligs

        # Loading from cache didn't work. Load from PDB file instead (slower).
        m = self._load_pdb(idx)

        ignore_sels = []
        lig_sels = []
        ligands = []

        for lig in self.ligands:
            # Note that "(altloc _ or altloc A)" makes sure only the first
            # alternate locations are used.
            lig_sel = f"chain {lig.chain} and resnum >= {lig.resnum} and resnum < {lig.resnum + lig.reslength} and (altloc _ or altloc A)"

            if lig.validity != "Part of Protein":
                ignore_sels.append(lig_sel)

            if lig.is_valid:
                lig_atoms = m.select(lig_sel)

                # Ligand may not be present in this biological assembly.
                if lig_atoms is None:
                    continue

                lig_mol = self._get_lig_from_prody_mol(lig_atoms, lig)

                if lig_mol is None:
                    continue

                ligands.append(lig_mol)
                lig_sels.append(lig_sel)

        rec_mol = self._get_rec_from_prody_mol(m, ignore_sels, lig_sels)

        self._save_to_file_cache(pkl_filename, rec_mol, ligands)

        return rec_mol, ligands


@dataclass
class MOAD_ligand(object):
    name: str
    validity: str
    affinity_measure: str
    affinity_value: str
    affinity_unit: str
    smiles: str

    @property
    def chain(self) -> str:
        return self.name.split(":")[1]

    @property
    def resnum(self) -> int:
        return int(self.name.split(":")[2])

    @property
    def reslength(self) -> int:
        return len(self.name.split(":")[0].split(" "))

    @property
    def is_valid(self) -> bool:
        return self.validity == "valid"


@dataclass
class MOAD_split(object):
    name: str
    targets: List[str]
    smiles: List[str]


@dataclass
class Entry_info(object):
    fragment_smiles: str
    parent_smiles: str  # TODO: Really need to calculate this? Not sure how long it takes, but not really needed.
    receptor_name: str
    connection_pt: List[float]

    def hashable_key(self) -> str:
        return (
            self.fragment_smiles
            + "--"
            + self.parent_smiles
            + "--"
            + self.receptor_name
            + "--"
            + str(self.connection_pt[0])
            + ","
            + str(self.connection_pt[1])
            + ","
            + str(self.connection_pt[2])
        )
