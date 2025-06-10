from dataclasses import dataclass
from typing import Any
from collagen.external.pdb_sdf_dir.targets_ligands import PdbSdfDir_target, PdbSdfDir_ligand


@dataclass
class PairedCsv_target(PdbSdfDir_target):
    """Class to load a target/ligand from PDB/SDF files. Identical to
    PdbSdfDir_target."""


@dataclass
class PairedCsv_ligand(PdbSdfDir_ligand):
    fragment_and_act: dict
    backed_parent: Any
    rdmol: Any
