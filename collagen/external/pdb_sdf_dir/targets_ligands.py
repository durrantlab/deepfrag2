from dataclasses import dataclass
from typing import Any, List, Tuple, Union
from collagen.core.molecules.mol import BackedMol, Mol
from collagen.external.common.parent_targets_ligands import Parent_ligand, Parent_target
import prody  # type: ignore


@dataclass
class PdbSdfDir_ligand(Parent_ligand):

    """Class to hold information about a ligand from a directory of PDB/SDF
    files.
    """

    rdmol: Any


@dataclass
class PdbSdfDir_target(Parent_target):
    """Class to load a target/ligand from a directory of PDB/SDF files."""

    def _get_lig_from_prody_mol(self, lig_atoms, lig):
        # This method might not be needed for PdbSdfDir_target, but we'll keep it for consistency
        pass

    def __getitem__(self, idx: int) -> Tuple[BackedMol, List[BackedMol]]:
        """Load the Nth structure for this target.

        Args:
            idx (int): The index of the biological assembly to load.

        Returns a (receptor, ligand) tuple of Mol objects.
        """
        lig_mols: List[BackedMol] = []

        for lig in self.ligands:
            # Make sure lig is a PdbSdfDir_ligand
            assert isinstance(lig, PdbSdfDir_ligand), "lig is not a PdbSdfDir_ligand"

            lig_mol = Mol.from_rdkit(lig.rdmol, strict=False)
            lig_mol.meta["name"] = lig.name
            lig_mol.meta["moad_ligand"] = lig
            lig_mols.append(lig_mol)

        rec_mol = None
        if self.pdb_id != "Non":
            m = self._load_pdb(idx)
            rec_mol = self._get_rec_from_prody_mol(m, [], [])
            rec_mol.meta["name"] = f"Receptor {self.pdb_id.lower()}"

        assert rec_mol is not None, "rec_mol is None"

        return rec_mol, lig_mols

    def _get_rec_from_prody_mol(
        self,
        m: Any,
        not_part_of_protein_sels: List[str],
        lig_sels: List[str],
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
