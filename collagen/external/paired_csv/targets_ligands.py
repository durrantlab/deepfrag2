from dataclasses import dataclass
from typing import Any

from collagen.external.common.parent_targets_ligands import Parent_ligand
from collagen.external.pdb_sdf_dir.targets_ligands import PdbSdfDir_target


@dataclass
class PairedCsv_target(PdbSdfDir_target):
    """Class to load a target/ligand from PDB/SDF files. Identical to
    PdbSdfDir_target."""


@dataclass
class PairedCsv_ligand(Parent_ligand):
    fragment_and_act: dict
    backed_parent: Any
    rdmol: Any
