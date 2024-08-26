"""Simple dataclasses like MOAD_class, MOAD_family, MOAD_target, etc. Note that
MOAD_target has some complexity to it (to load/save PDB files, including
caching), but let's leave it here.
"""

from dataclasses import dataclass, field
from collagen.core import args as user_args
from typing import List, Tuple, Any
from collections import OrderedDict
from pathlib import Path
import textwrap
from collagen.core.molecules.mol import (
    TemplateGeometryMismatchException,
    UnparsableGeometryException,
    UnparsableSMILESException,
)
import pickle
from io import StringIO
import prody
from ... import Mol
from .moad_utils import fix_moad_smiles
import sys
import os


@dataclass
class MOAD_class(object):

    """MOAD class."""

    ec_num: str
    families: List["MOAD_family"]


@dataclass
class MOAD_family(object):

    """MOAD family."""

    rep_pdb_id: str
    targets: List["MOAD_target"]


@dataclass
class MOAD_target(object):

    """MOAD target."""

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

    recent_pickle_contents = OrderedDict()

    def __len__(self) -> int:
        """Return the number of on-disk structures."""
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

    def _get_lig_from_prody_mol(self, lig_atoms, lig):
        try:
            lig_mol = Mol.from_prody(
                lig_atoms, fix_moad_smiles(lig.smiles), sanitize=True
            )

            lig_mol.meta["name"] = lig.name
            lig_mol.meta["moad_ligand"] = lig

            return lig_mol

        # Catch a whole bunch of errors.
        except UnparsableSMILESException as err:
            if user_args.verbose:
                msg = str(err).replace("[LIGAND]", f"{self.pdb_id}:{lig.name}")
                print("\n", file=sys.stderr)
                print(textwrap.fill(msg, subsequent_indent="  "), file=sys.stderr)
            return None
        except UnparsableGeometryException as err:
            if user_args.verbose:
                msg = str(err).replace("[LIGAND]", f"{self.pdb_id}:{lig.name}")
                print("\n", file=sys.stderr)
                print(textwrap.fill(msg, subsequent_indent="  "), file=sys.stderr)
            return None
        except TemplateGeometryMismatchException as err:
            if user_args.verbose:
                msg = str(err).replace("[LIGAND]", f"{self.pdb_id}:{lig.name}")
                print("\n", file=sys.stderr)
                print(textwrap.fill(msg, subsequent_indent="  "), file=sys.stderr)
            return None
        except Exception as err:
            if user_args.verbose:
                msg = f"\nWARNING: Could not process ligand {self.pdb_id}:{lig.name}. An unknown error occurred: {str(err)}"
                print(textwrap.fill(msg, subsequent_indent="  "), file=sys.stderr)
            return None

    def _get_rec_from_prody_mol(
        self,
        m: Any,
        not_part_of_protein_sels: List[str],
        lig_sels: List[str],
        # debug=False,
    ):
        # not_protein_sels contains the selections of ligands that are not
        # considered part of the receptor (such as cofactors). These shouldn't
        # be included in the protein selection.
        if not_part_of_protein_sels:
            rec_sel = "not water and not (%s)" % " or ".join(
                f"({x})" for x in not_part_of_protein_sels
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
            dist = (3 ** 0.5) * dist

            # Add padding
            dist = dist + self.grid_padding

            rec_sel = f"{rec_sel} and exwithin {str(dist)} of ({all_lig_sel})"

        if self.noh:
            # Removing hydrogen atoms (when not needed) also speeds the
            # calculations.
            rec_sel = f"not hydrogen and {rec_sel}"

        # if debug:
        #     print("1", "rec_sel", rec_sel)
        #     # print("2", m.select(rec_sel))
        #     # print("3", Mol.from_prody(m.select(rec_sel)))

        # Note that "(altloc _ or altloc A)" makes sure only the first alternate
        # locations are used.
        rec_sel = f"{rec_sel} and (altloc _ or altloc A)"

        try:
            # So strange. Sometimes prody can't parse perfectly valid selection
            # strings, but if you just try a second time, it works. I don't know
            # why.
            prody_mol = m.select(rec_sel)
        except prody.atomic.select.SelectionError as e:
            prody_mol = m.select(rec_sel)
            # import pdb; pdb.set_trace()

        # Print numer of atoms in selection
        # if prody_mol is None:
        # print(rec_sel)
        # print("Number of atoms in selection:", prody_mol.numAtoms())
        rec_mol = Mol.from_prody(prody_mol)
        rec_mol.meta["name"] = f"Receptor {self.pdb_id.lower()}"

        return rec_mol

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

        not_part_of_protein_sels = []
        lig_sels = []
        lig_mols = []

        for lig in self.ligands:
            # Get the selection of the ligand. Accounts for multi-residue
            # ligands (>=, <).
            lig_sel = f"chain {lig.chain} and "
            lig_sel += (
                f"resnum {lig.resnum}"
                if lig.reslength <= 1
                else f"resnum >= {lig.resnum} and resnum < {lig.resnum + lig.reslength}"
            )
            # lig_sel += " and (altloc _ or altloc A)"

            # Save a list of those ligand selections that are not considered
            # part of the protein (e.g., cofactors, metals). You'll use these
            # later when you're getting the receptor atoms.
            if lig.validity != "Part of Protein":
                not_part_of_protein_sels.append(lig_sel)

            # If it's a valid ligand, make a prody mol from it.
            if lig.is_valid:
                # Always add the selection. This is used to get the protein
                # around all ligands.
                lig_sels.append(lig_sel)

                # Note that "(altloc _ or altloc A)" makes sure only the first
                # alternate locations are used.
                try:
                    lig_atoms = m.select(f"{lig_sel} and (altloc _ or altloc A)")
                except prody.atomic.select.SelectionError as e:
                    # So strange. Sometimes prody can't parse perfectly valid
                    # selection strings, but if you just try a second time, it
                    # works. I don't know why. Related to Cesar multiprocessing
                    # method somehow (unsure)?
                    lig_atoms = m.select(f"{lig_sel} and (altloc _ or altloc A)")

                # Ligand may not be present in this biological assembly.
                if lig_atoms is None:
                    continue

                lig_mol = self._get_lig_from_prody_mol(lig_atoms, lig)

                if lig_mol is None:
                    continue

                lig_mols.append(lig_mol)
                # lig_sels.append(lig_sel)

        # Sort the selection lists to help with debugging
        # lig_sels.sort()
        # not_part_of_protein_sels.sort()

        # Now make a prody mol for the receptor.
        rec_mol = self._get_rec_from_prody_mol(m, not_part_of_protein_sels, lig_sels)

        # print(pkl_filename)
        self._save_to_file_cache(pkl_filename, rec_mol, lig_mols)

        return rec_mol, lig_mols


@dataclass
class PdbSdfDir_target(MOAD_target):

    """Class to load a target/ligand from a directory of PDB/SDF files."""

    def __getitem__(self, idx: int) -> Tuple[Mol, Mol]:
        """Load the Nth structure for this target.

        Args:
            idx (int): The index of the biological assembly to load.

        Returns a (receptor, ligand) tuple of Mol objects.
        """
        lig_mols = []

        for lig in self.ligands:
            lig_mol = Mol.from_rdkit(lig.rdmol, strict=False)
            lig_mol.meta["name"] = lig.name
            lig_mol.meta["moad_ligand"] = lig
            lig_mols.append(lig_mol)

        rec_mol = None
        if self.pdb_id != "Non":
            m = self._load_pdb(idx)
            rec_mol = self._get_rec_from_prody_mol(m, [], [])
            rec_mol.meta["name"] = f"Receptor {self.pdb_id.lower()}"

        return rec_mol, lig_mols

    def _get_rec_from_prody_mol(
        self, m: Any, not_part_of_protein_sels: List[str], lig_sels: List[str],
    ):
        """Get the receptor atoms from a prody mol.

        Args:
            m (Any): A prody mol.
            not_part_of_protein_sels (List[str]): A list of selections for
                components that are not part of the protein.
            lig_sels (List[str]): A list of selections for the ligands.

        Returns:
            Mol: The receptor atoms.
        """
        rec_sel = "not water"

        if self.noh:
            # Removing hydrogen atoms (when not needed) also speeds the
            # calculations.
            rec_sel = f"not hydrogen and {rec_sel}"

        # Note that "(altloc _ or altloc A)" makes sure only the first alternate
        # locations are used.
        rec_sel = f"{rec_sel} and (altloc _ or altloc A)"

        try:
            # So strange. Sometimes prody can't parse perfectly valid selection
            # strings, but if you just try a second time, it works. I don't know
            # why.
            prody_mol = m.select(rec_sel)
        except prody.atomic.select.SelectionError as e:
            prody_mol = m.select(rec_sel)
            # import pdb; pdb.set_trace()

        rec_mol = Mol.from_prody(prody_mol)
        rec_mol.meta["name"] = f"Receptor {self.pdb_id.lower()}"

        return rec_mol


@dataclass
class MOAD_ligand(object):

    """Class to hold information about a ligand from the MOAD database."""

    name: str
    validity: str
    affinity_measure: str
    affinity_value: str
    affinity_unit: str
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


@dataclass
class PdbSdfDir_ligand(MOAD_ligand):

    """Class to hold information about a ligand from a directory of PDB/SDF
    files.
    """

    rdmol: Any


@dataclass
class PairedPdbSdfCsv_ligand(PdbSdfDir_ligand):
    fragment_and_act: {}
    backed_parent: Any


@dataclass
class MOAD_split(object):

    """Class to hold information about a split of the MOAD database."""

    name: str
    targets: List[str]
    smiles: List[str]


@dataclass
class Entry_info(object):

    """Class to hold information about a MOAD entry."""

    fragment_smiles: str

    # TODO: Really need to calculate this? Not sure how long it takes, but not
    # really needed.
    parent_smiles: str

    receptor_name: str
    connection_pt: List[float]

    # these attributes are used when performing fine-tuning on paired data
    ligand_id: str
    fragment_idx: int

    def hashable_key(self) -> str:
        """Get a hashable key for this entry.

        Returns:
            str: A hashable key for this entry.
        """
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
