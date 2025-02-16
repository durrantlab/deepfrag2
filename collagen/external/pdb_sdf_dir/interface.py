from pathlib import Path
from typing import Union
import os
from collagen.external.common.parent_interface import ParentInterface
from collagen.external.common.types import StructuresClass, StructuresFamily
from collagen.external.pdb_sdf_dir.targets_ligands import (
    PdbSdfDir_ligand,
    PdbSdfDir_target,
)
from rdkit import Chem  # type: ignore
import linecache
import pandas as pd


class PdbSdfDirInterface(ParentInterface):

    """Interface for data stored in a directory of PDBs and SDFs."""

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
        """Interface for data stored in a directory of PDBs and SDFs.

        Args:
            metadata (Union[str, Path]): CSV file containing the PDB and SDF files.
            structures_path (Union[str, Path]): Path to the directory containing the
                PDBs and SDFs.
            cache_pdbs_to_disk (bool): Whether to cache the PDBs to disk.
            grid_width (int): Width of the grid.
            grid_resolution (float): Resolution of the grid.
            noh (bool): Whether to remove hydrogens.
            discard_distant_atoms (bool): Whether to discard distant atoms.
        """
        super().__init__(
            metadata,
            structures_path,
            cache_pdbs_to_disk,
            grid_width,
            grid_resolution,
            noh,
            discard_distant_atoms,
        )

    def _load_targets_ligands_hierarchically(
        self,
        metadata: Union[str, Path],
        cache_pdbs_to_disk: bool,
        grid_width: int,
        grid_resolution: float,
        noh: bool,
        discard_distant_atoms: bool,
    ):
        """Load the classes, families, targets, and ligands from the
        CSV file.

        Args:
            dir_path (Union[str, Path]): Path to the directory containing PDB
                and SDF files.
            cache_pdbs_to_disk (bool): Whether to cache the PDBs to disk.
            grid_width (int): Width of the grid.
            grid_resolution (float): Resolution of the grid.
            noh (bool): Whether to remove hydrogens.
            discard_distant_atoms (bool): Whether to discard distant atoms.
        """
        receptor_ligand_pairs = {}
        df = pd.read_csv(metadata)
        for index, row in df.iterrows():
            pdb_path = row['receptor']
            if pdb_path not in receptor_ligand_pairs:
                receptor_ligand_pairs[pdb_path] = []
            receptor_ligand_pairs[pdb_path].append(row['ligand'])

        classes = []
        curr_class = None
        curr_class_name = None
        curr_family = None
        curr_family_name = None
        curr_target = None
        curr_target_name = None

        pdb_files = list(receptor_ligand_pairs)
        pdb_files.sort()
        for pdb_path in pdb_files:
            pdb_path_parts = pdb_path.split(os.sep)
            full_pdb_name = pdb_path_parts[len(pdb_path_parts) - 1].split(".")[0]

            if (curr_target is None) or (full_pdb_name != curr_target_name):
                if curr_target is not None and curr_family is not None:
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
                for sdf_path in receptor_ligand_pairs[pdb_path]:
                    sdf_reader = Chem.SDMolSupplier(sdf_path)
                    for ligand_ in sdf_reader:
                        if ligand_ is not None:
                            curr_target.ligands.append(
                                PdbSdfDir_ligand(
                                    name=linecache.getline(sdf_path, 1).rstrip(),
                                    validity="valid",
                                    # affinity_measure="",
                                    # affinity_value="",
                                    # affinity_unit="",
                                    smiles=Chem.MolToSmiles(ligand_),
                                    rdmol=ligand_,
                                )
                            )

            if (curr_family is None) or (full_pdb_name != curr_family_name):
                if curr_family is not None and curr_class is not None:
                    curr_class.families.append(curr_family)
                curr_family_name = full_pdb_name
                curr_family = StructuresFamily(rep_pdb_id=full_pdb_name, targets=[])

            if curr_class is None:
                curr_class_name = full_pdb_name
                curr_class = StructuresClass(ec_num=full_pdb_name, families=[])
            elif full_pdb_name != curr_class_name:
                classes.append(curr_class)
                curr_class_name = full_pdb_name
                curr_class = StructuresClass(ec_num=full_pdb_name, families=[])

        if curr_target is not None and curr_family is not None:
            curr_family.targets.append(curr_target)
        if curr_family is not None and curr_class is not None:
            curr_class.families.append(curr_family)
        if curr_class is not None:
            classes.append(curr_class)

        self.classes = classes

    def _get_structure_file_extension(self) -> Union[str, None]:
        return "pdb"
