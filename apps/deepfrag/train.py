import argparse
from typing import List, Tuple

import torch
from torch.utils import data

from collagen import Mol, DelayedMolVoxel, VoxelParams
from collagen.external.moad import MOADFragmentDataset
from collagen.util import rand_rot
from collagen.skeletons import MoadVoxelSkeleton
from collagen.core.args import get_args

from model import DeepFragModel


ENTRY_T = Tuple[Mol, Mol, Mol]
TMP_T = Tuple[DelayedMolVoxel, DelayedMolVoxel, torch.Tensor]
OUT_T = Tuple[torch.Tensor, torch.Tensor]


class DeepFrag(MoadVoxelSkeleton):
    def __init__(self):
        super().__init__(
            model_cls=DeepFragModel, dataset_cls=MOADFragmentDataset,
        )

    @staticmethod
    def pre_voxelize(
        args: argparse.Namespace, voxel_params: VoxelParams, entry: ENTRY_T
    ) -> TMP_T:
        rec, parent, frag = entry
        rot = rand_rot()
        return (
            rec.voxelize_delayed(voxel_params, center=frag.connectors[0], rot=rot),
            parent.voxelize_delayed(voxel_params, center=frag.connectors[0], rot=rot),
            torch.tensor(frag.fingerprint("rdk10", args.fp_size)),
        )

    @staticmethod
    def voxelize(
        args: argparse.Namespace,
        voxel_params: VoxelParams,
        device: torch.device,
        batch: List[TMP_T],
    ) -> OUT_T:
        voxels = torch.zeros(
            size=voxel_params.tensor_size(batch=len(batch), feature_mult=2),
            device=device,
        )

        fingerprints = torch.zeros(size=(len(batch), args.fp_size), device=device)

        for i in range(len(batch)):
            rec, parent, frag = batch[i]

            rec.voxelize_into(
                voxels, batch_idx=i, layer_offset=0, cpu=(device.type == "cpu")
            )

            parent.voxelize_into(
                voxels,
                batch_idx=i,
                layer_offset=voxel_params.atom_featurizer.size(),
                cpu=(device.type == "cpu"),
            )

            fingerprints[i] = frag

        return (voxels, fingerprints)


if __name__ == "__main__":
    args = get_args(
        parser_funcs=[MoadVoxelSkeleton.add_moad_args, DeepFragModel.add_model_args],
        fix_args_funcs=[MoadVoxelSkeleton.fix_moad_args],
        is_pytorch_lightning=True,
    )
    DeepFrag().run(args)
