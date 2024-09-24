"""Simple dataclasses like StructuresClass, StructuresFamily, MOAD_target, etc. Note that
MOAD_target has some complexity to it (to load/save PDB files, including
caching), but let's leave it here.
"""

from dataclasses import dataclass
from collagen.core import args as user_args
from typing import List, Tuple, Any

import textwrap
from collagen.core.molecules.mol import (
    BackedMol,
    TemplateGeometryMismatchException,
    UnparsableGeometryException,
    UnparsableSMILESException,
)

from collagen.external.common.parent_targets_ligands import Parent_ligand, Parent_target
import prody  # type: ignore
from collagen.core.molecules.mol import Mol
from collagen.external.common.utils import fix_smiles
import sys


@dataclass
class MOAD_target(Parent_target):
    """MOAD target."""

    def _get_lig_from_prody_mol(self, lig_atoms, lig):
        try:
            lig_mol = Mol.from_prody(
                lig_atoms, fix_smiles(lig.smiles), sanitize=True
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
            dist = (3**0.5) * dist

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

    def __getitem__(self, idx: int) -> Tuple[BackedMol, List[BackedMol]]:
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
        lig_mols: List[BackedMol] = []

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
class MOAD_ligand(Parent_ligand):
    """Class to hold information about a ligand from the MOAD database."""

    affinity_measure: str
    affinity_value: str
    affinity_unit: str
