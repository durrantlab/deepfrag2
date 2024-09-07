from pathlib import Path
from typing import Union
import glob
import os
from collagen.external.common.parent_interface import ParentInterface
from collagen.external.common.types import StructuresClass, StructuresFamily
from collagen.external.pdb_sdf_dir.targets_ligands import (
    PdbSdfDir_ligand,
    PdbSdfDir_target,
)
from rdkit import Chem  # type: ignore
import linecache


class PdbSdfDirInterface(ParentInterface):

    """Interface for data stored in a directory of PDBs and SDFs."""

    def __init__(
        self,
        structures_dir: Union[str, Path],
        cache_pdbs_to_disk: bool,
        grid_width: int,
        grid_resolution: float,
        noh: bool,
        discard_distant_atoms: bool,
    ):
        """Interface for data stored in a directory of PDBs and SDFs.

        Args:
            structures_dir (Union[str, Path]): Path to the directory containing the
                PDBs and SDFs.
            cache_pdbs_to_disk (bool): Whether to cache the PDBs to disk.
            grid_width (int): Width of the grid.
            grid_resolution (float): Resolution of the grid.
            noh (bool): Whether to remove hydrogens.
            discard_distant_atoms (bool): Whether to discard distant atoms.
        """
        super().__init__(
            structures_dir,
            structures_dir,
            cache_pdbs_to_disk,
            grid_width,
            grid_resolution,
            noh,
            discard_distant_atoms,
        )

    def _load_targets_ligands_hierarchically(
        self,
        dir_path: Union[str, Path],
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
        classes = []
        curr_class = None
        curr_class_name = None
        curr_family = None
        curr_family_name = None
        curr_target = None
        curr_target_name = None

        # if every_csv_path is a Path, convert to str
        if isinstance(dir_path, Path):
            dir_path = str(dir_path)

        pdb_files = glob.glob(dir_path + os.sep + "*.pdb", recursive=True)
        pdb_files.sort()
        for line in pdb_files:
            parts = line.split(os.sep)
            full_pdb_name = parts[len(parts) - 1].split(".")[0]
            parts = full_pdb_name.split("_")

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
                sdf_name = f"{parts[0]}_lig_{parts[2]}"
                sdf_reader = Chem.SDMolSupplier(
                    dir_path + os.sep + sdf_name + ".sdf"
                )
                for ligand_ in sdf_reader:
                    # it is expected only one iteration because each SDF file
                    # must have a single molecule
                    if ligand_ is not None:
                        curr_target.ligands.append(
                            PdbSdfDir_ligand(
                                name=linecache.getline(
                                    dir_path + os.sep + sdf_name + ".sdf", 1
                                ).rstrip(),
                                validity="valid",
                                # affinity_measure="",
                                # affinity_value="",
                                # affinity_unit="",
                                smiles=Chem.MolToSmiles(ligand_),
                                rdmol=ligand_,
                            )
                        )

            if (curr_family is None) or (parts[0] != curr_family_name):
                if curr_family is not None and curr_class is not None:
                    curr_class.families.append(curr_family)
                curr_family_name = parts[0]
                curr_family = StructuresFamily(rep_pdb_id=parts[0], targets=[])

            if curr_class is None:
                curr_class_name = parts[0]
                curr_class = StructuresClass(ec_num=parts[0], families=[])
            elif parts[0] != curr_class_name:
                classes.append(curr_class)
                curr_class_name = parts[0]
                curr_class = StructuresClass(ec_num=parts[0], families=[])

        if curr_target is not None and curr_family is not None:
            curr_family.targets.append(curr_target)
        if curr_family is not None and curr_class is not None:
            curr_class.families.append(curr_family)
        if curr_class is not None:
            classes.append(curr_class)

        self.classes = classes

    def _get_structure_file_extension(self) -> Union[str, None]:
        return "pdb"
