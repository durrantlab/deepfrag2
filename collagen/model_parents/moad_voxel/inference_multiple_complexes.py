"""A model for inference on a custom dataset."""

from collagen.external.common.parent_interface import ParentInterface
from collagen.external.pdb_sdf_dir.interface import PdbSdfDirInterface
from argparse import Namespace
import torch  # type: ignore
from typing import Any, Tuple, Optional
from collagen.core.voxelization.voxelizer import VoxelParams
from collagen.model_parents.moad_voxel.inference import Inference


class InferenceMultipleComplex(Inference):

    """A model for inference on a custom set."""

    def __init__(self, model_parent: Any):
        """Initialize the model.

        Args:
            model_parent (Any): The parent model.
        """
        Inference.__init__(self, model_parent)

    def _validate_run_test(self, args: Namespace, ckpt_filename: Optional[str]):
        """Validate the arguments required to run inference.

        Args:
            args (Namespace): The arguments.
            ckpt_filename (Optional[str]): The checkpoint.

        Raises:
            ValueError: If the arguments are invalid.
        """
        super()._validate_run_test(args, ckpt_filename)

        if not args.csv_of_receptor_ligand_complexes or not args.path_to_receptor_ligand_complexes:
            raise Exception(
                "Specify the --csv_of_receptor_ligand_complexes and --path_to_receptor_ligand_complexes parameters. "
                "These parameters contain the paths to the receptor-ligand complexes to be used in the inference mode."
            )

    def _read_datasets_to_run_test(
        self, args: Namespace, voxel_params: VoxelParams
    ) -> Tuple[PdbSdfDirInterface, ParentInterface]:
        """Read the datasets required to run inference.

        Args:
            args (Namespace): The arguments.
            voxel_params (VoxelParams): The voxel parameters.

        Returns:
            Tuple[PdbSdfDirInterface, MOADInterface]: PdbSdfDirInterface contains the dataset to run test on,
                whereas MOADInterface contains the fragments to be used in the prediction.
        """
        print("Loading MOAD database.")
        moad = None
        if args.every_csv and args.data_dir:
            moad = self._read_BindingMOAD_database(args, voxel_params)

        print("Loading custom database.")
        dataset = PdbSdfDirInterface(
            metadata=args.csv_of_receptor_ligand_complexes,
            structures_path=args.path_to_receptor_ligand_complexes,
            cache_pdbs_to_disk=args.cache_pdbs_to_disk,
            grid_width=voxel_params.width,
            grid_resolution=voxel_params.resolution,
            noh=args.noh,
            discard_distant_atoms=args.discard_distant_atoms,
        )

        return dataset, moad
