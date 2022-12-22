from argparse import Namespace
from typing import Any, List, Optional
from collagen.checkpoints import get_last_checkpoint
from collagen.core.loader import DataLambda, MultiLoader
from collagen.core.molecules.mol import Mol
from collagen.core.voxelization.voxelizer import VoxelParams
import torch
from collagen.external.moad.interface import MOADInterface
from collagen.external.moad.types import MOAD_split


class MoadVoxelModelUtils(object):

    @staticmethod
    def disable_warnings():
        from rdkit import RDLogger
        import prody

        RDLogger.DisableLog("rdApp.*")
        prody.confProDy(verbosity="none")

    def get_data_from_split(
        self: "MoadVoxelModelParent",
        cache_file: str,
        args: Namespace,
        dataset: MOADInterface,
        split: MOAD_split,
        voxel_params: VoxelParams,
        device: Any,
        shuffle=True,
    ) -> DataLambda:
        # This is where you do actual dataset construction. The transform
        # function actually gets the data (voxelizes and creates fingerprint).
        # Note also call to self.dataset_cls should create the .json.cache file.
        # TODO: Create separate function .pre_voxelize_with_voxel that just
        # calculates just fingerprint.

        # JDD NOTE: self.dataset_cls could be something like MOADFragmentDataset

        dataset = self.dataset_cls(
            moad=dataset,
            cache_file=cache_file,
            cache_cores=args.num_dataloader_workers,
            split=split,
            transform=(
                lambda entry: self.__class__.pre_voxelize(args, voxel_params, entry)
            ),
            args=args,
        )
        data = (
            MultiLoader(
                dataset,
                shuffle=shuffle,
                num_dataloader_workers=args.num_dataloader_workers,
                max_voxels_in_memory=args.max_voxels_in_memory,
            ).batch(args.batch_size)
            # self.__class__.voxelize below actually makes the voxel and
            # fragments, I think. For fragments only, use
            # voxel_params.frag_fpts_only = True
            .map(lambda batch: self.__class__.voxelize(args, voxel_params, device, batch))
        )

        return data

    @staticmethod
    def get_checkpoint(args: Namespace) -> Optional[str]:
        if args.load_checkpoint and args.load_newest_checkpoint:
            raise ValueError(
                "Can specify 'load_checkpoint=xyz' or 'load_newest_checkpoint' but not both."
            )

        if args.model_for_warm_starting and (args.load_checkpoint or args.load_newest_checkpoint):
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

    @staticmethod
    def debug_smis_match_fps(fps: torch.Tensor, smis: List[str], device: Any):
        import rdkit
        from rdkit import Chem

        for idx in range(len(smis)):
            smi = smis[idx]
            fp1 = fps[idx]

            mol = Mol.from_smiles(smi)
            # TODO: 2048 should be hardcoded here? I think it's a user parameter.
            fp2 = torch.tensor(
                mol.fingerprint("rdk10", 2048), device=device, dtype=torch.float32
            )
            print((fp1 - fp2).max() == (fp1 - fp2).min())

        import pdb; pdb.set_trace()
