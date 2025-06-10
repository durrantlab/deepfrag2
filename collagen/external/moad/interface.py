"""Interface for Binding MOAD data."""

from pathlib import Path
from typing import Union

from collagen.external.common.parent_interface import ParentInterface
from collagen.external.common.types import StructuresClass, StructuresFamily
from .targets_ligands import (
    MOAD_ligand,
    MOAD_target,
)


class MOADInterface(ParentInterface):

    """Base class for interacting with Binding MOAD data. Initialize by passing
    the path to "every.csv" and the path to a folder containing structure files
    (can be nested).

    NOTE: This just interfaces with the BindingMOAD on disk. It doesn't
    fragment those ligands (see fragment_dataset.py). It doesn't calculate the
    properties of the ligands/fragments or filter them (see cache_filter.py).

    Args:
        metadata: Path to the metadata "every.csv" file.
        structures: Path to a folder container structure files.
    """

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
        """Initialize a MOADInterface object.

        Args:
            metadata (Union[str, Path]): Path to the metadata "every.csv" file.
            structures_path (Union[str, Path]): Path to a folder container structure files.
            cache_pdbs_to_disk (bool): Whether to cache PDBs to disk.
            grid_width (int): Grid width.
            grid_resolution (float): Grid resolution.
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
        csv_path: Union[str, Path],
        cache_pdbs_to_disk: bool,
        grid_width: int,
        grid_resolution: float,
        noh: bool,
        discard_distant_atoms: bool,
    ):
        """BindingMOAD data is loaded into protein classes, which contain
        families, which contain the individual targets, which are associated
        with ligands. This function sets up a heirarchical data structure that
        preserves these relationships. The structure is comprised of nested
        StructuresClass, StructuresFamily, MOAD_target, and MOAD_ligand dataclasses.

        Args:
            csv_path (Union[str, Path]): Path to the metadata "every.csv"
                file.
            cache_pdbs_to_disk (bool): Whether to cache PDBs to disk.
            grid_width (int): Grid width.
            grid_resolution (float): Grid resolution.
            noh (bool): Whether to remove hydrogens.
            discard_distant_atoms (bool): Whether to discard distant atoms.
        """
        # Note that the output of this function gets put in self.classes.

        with open(csv_path, "r") as f:
            dat = f.read().strip().split("\n")

        classes = []
        curr_class = None
        curr_family = None
        curr_target = None

        for line in dat:
            parts = line.split(",")

            if parts[0] != "":  # 1: Protein Class
                if curr_class is not None:
                    classes.append(curr_class)
                curr_class = StructuresClass(ec_num=parts[0], families=[])
            elif parts[1] != "":  # 2: Protein Family
                if curr_target is not None and curr_family is not None:
                    curr_family.targets.append(curr_target)
                if curr_family is not None and curr_class is not None:
                    curr_class.families.append(curr_family)
                curr_family = StructuresFamily(rep_pdb_id=parts[2], targets=[])
                curr_target = MOAD_target(
                    pdb_id=parts[2],
                    ligands=[],
                    cache_pdbs_to_disk=cache_pdbs_to_disk,
                    grid_width=grid_width,
                    grid_resolution=grid_resolution,
                    noh=noh,
                    discard_distant_atoms=discard_distant_atoms,
                )
            elif parts[2] != "":  # 3: Protein target
                if curr_target is not None and curr_family is not None:
                    curr_family.targets.append(curr_target)

                # if "-" in parts[2]:
                # print(parts[2], "+++++")
                # logit(f"Loading {parts[2]}", "~/work_dir/make_moad_target.txt")

                curr_target = MOAD_target(
                    pdb_id=parts[2],
                    ligands=[],
                    cache_pdbs_to_disk=cache_pdbs_to_disk,
                    grid_width=grid_width,
                    grid_resolution=grid_resolution,
                    noh=noh,
                    discard_distant_atoms=discard_distant_atoms,
                )
            elif (
                parts[3] != ""
                and curr_target is not None
                and curr_target.ligands is not None
            ):  # 4: Ligand
                curr_target.ligands.append(
                    MOAD_ligand(
                        name=parts[3],
                        validity=parts[4],
                        affinity_measure=parts[5],
                        affinity_value=parts[7],
                        affinity_unit=parts[8],
                        smiles=parts[9],
                    )
                )

        if curr_target is not None and curr_family is not None:
            curr_family.targets.append(curr_target)
        if curr_family is not None and curr_class is not None:
            curr_class.families.append(curr_family)
        if curr_class is not None:
            classes.append(curr_class)

        self.classes = classes

    def _get_structure_file_extension(self) -> Union[str, None]:
        return "bio*"
