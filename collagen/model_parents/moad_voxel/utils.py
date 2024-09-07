"""Utilities for MOAD voxel model."""

from argparse import Namespace
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
from collagen.checkpoints import get_last_checkpoint
from collagen.core.loader import DataLambda, MultiLoader
from collagen.core.voxelization.voxelizer import VoxelParams
from collagen.external.common.parent_interface import ParentInterface
from collagen.external.common.types import StructuresSplit
from collagen.model_parents.moad_voxel.moad_voxel import VoxelModelParent
import torch  # type: ignore


class VoxelModelUtils(object):
    """Provides utility funcitons."""

    def __init__(self, parent: "VoxelModelParent"):
        """Initialize the class.

        Args:
            parent (VoxelModelParent): The parent class.
        """
        self.parent = parent

    @staticmethod
    def disable_warnings():
        """Disable warnings from RDKit and ProDy."""
        from rdkit import RDLogger  # type: ignore
        import prody  # type: ignore

        RDLogger.DisableLog("rdApp.*")
        prody.confProDy(verbosity="none")

    def get_data_from_split(
        self,
        cache_file: str,
        args: Namespace,
        data_interface: ParentInterface,
        voxel_params: VoxelParams,
        device: torch.device,
        # NOTE: `split` needs to be optional because no split when running
        # inference. See inference_custom_dataset.py.
        split: Optional[StructuresSplit],
        shuffle=True,
    ) -> DataLambda:
        """Where you do actual dataset construction. The transform
        function actually gets the data (voxelizes and creates fingerprint).
        Note also call to self.dataset_cls should create the .json.cache file.
        TODO: Create separate function .pre_voxelize_with_voxel that just
        calculates just fingerprint.

        Args:
            self: This object
            cache_file (str): The cache file to use.
            args (Namespace): The user arguments.
            data_interface (ParentInterface): The dataset interface.
            voxel_params (VoxelParams): Parameters for voxelization.
            device (torch.device): The device to use.
            split (Optional[StructuresSplit]): The split to use.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

        Returns:
            DataLambda: The data.
        """
        # JDD NOTE: self.dataset_cls could be something like MOADFragmentDataset

        dataset = self.parent.dataset_cls(
            data_interface=data_interface,
            cache_file=cache_file,
            cache_cores=args.num_dataloader_workers,
            split=split,
            transform=(
                lambda entry: self.parent.__class__.pre_voxelize(
                    args, voxel_params, entry
                )
            ),
            args=args,
        )

        return (
            MultiLoader(
                dataset,
                args.fragment_representation,
                shuffle=shuffle,
                num_dataloader_workers=args.num_dataloader_workers,
                max_voxels_in_memory=args.max_voxels_in_memory,
            ).batch(args.batch_size)
            # self.__class__.voxelize below actually makes the voxel and
            # fragments, I think. For fragments only, use
            # voxel_params.frag_fpts_only = True
            .map(
                lambda batch: self.parent.__class__.voxelize(
                    args, voxel_params, device, batch
                )
            )
        )

    @staticmethod
    def get_checkpoint_filename(
        args: Namespace, validate_args: bool = True
    ) -> Optional[str]:
        """Get the checkpoint filename.

        Args:
            args (Namespace): The arguments parsed by argparse.
            validate_args (bool, optional): Whether to validate the arguments.
                Defaults to True.

        Raises:
            ValueError: If the arguments are invalid.

        Returns:
            Optional[str]: The checkpoint filename.
        """
        if validate_args:
            if args.load_checkpoint and args.load_newest_checkpoint:
                raise ValueError(
                    "Can specify 'load_checkpoint=xyz' or 'load_newest_checkpoint' but not both."
                )

            if args.model_for_warm_starting and (
                args.load_checkpoint or args.load_newest_checkpoint
            ):
                raise ValueError(
                    "If warm starting will be performed, then it cannot specify 'load_checkpoint=xyz' nor 'load_newest_checkpoint'."
                )
            if args.mode == "warm_starting" and not args.model_for_warm_starting:
                raise ValueError(
                    "If 'warm_starting' mode was specified, then it must be specified the 'model_for_warm_starting' parameter."
                )
            if args.model_for_warm_starting and args.mode != "warm_starting":
                raise ValueError(
                    "The 'model_for_warm_starting' parameter is only valid when 'warm_starting' mode is specified."
                )

        ckpt = None
        if args.load_checkpoint:
            ckpt = args.load_checkpoint
        elif args.load_newest_checkpoint:
            ckpt = get_last_checkpoint(args)

        return ckpt

    # def debug_smis_match_fps(
    #     self, smis: List[str], device: torch.device, args: Namespace
    # ):
    #     """Debugging function to check that the SMILES strings match the
    #     fingerprints.

    #     Args:
    #         smis (List[str]): The SMILES strings.
    #         device (torch.device): The device to use.
    #         args (Namespace): The user arguments.
    #     """
    #     # TODO: Not currently used.

    #     for idx in range(len(smis)):
    #         smi = smis[idx]
    #         fp1 = self[idx]

    #         mol = Mol.from_smiles(smi)
    #         # TODO: 2048 should be hardcoded here? I think it's a user parameter.
    #         fp2 = torch.tensor(
    #             mol.fingerprint(args.fragment_representation, args.fp_size),
    #             device=device,
    #             dtype=torch.float32,
    #         )
    #         print((fp1 - fp2).max() == (fp1 - fp2).min())

    #     import pdb

    #     pdb.set_trace()
