
import argparse
from typing import List, Tuple

import torch

from collagen import Mol, DelayedMolVoxel, VoxelParams
from collagen.external.moad import MOADPocketDataset
from collagen.util import rand_rot
from collagen.skeletons import MoadVoxelSkeleton

from model import BindingSiteModel


ENTRY_T = Tuple[Mol, "numpy.ndarray", "numpy.ndarray"] # (rec, pos_coord, neg_coord)
TMP_T = Tuple[DelayedMolVoxel, DelayedMolVoxel] # (pos_voxel, neg_voxel)
OUT_T = Tuple[torch.Tensor, torch.Tensor] # (voxels, targets)


class BindingSiteVoxel(MoadVoxelSkeleton):
    def __init__(self):
        super().__init__(
            model_cls=BindingSiteModel,
            dataset_cls=MOADPocketDataset,
        )

    @staticmethod
    def pre_voxelize(args: argparse.Namespace, voxel_params: VoxelParams, entry: ENTRY_T) -> TMP_T:
        rec, pos, neg = entry
        rot = rand_rot()
        return (
            rec.voxelize_delayed(voxel_params, center=pos, rot=rot),
            rec.voxelize_delayed(voxel_params, center=neg, rot=rot),
        )
    @staticmethod
    def voxelize(args: argparse.Namespace, voxel_params: VoxelParams, device: torch.device, batch: List[TMP_T]) -> OUT_T:
        data = [x for x in batch if x is not None]
        if len(data) == 0:
            return None

        voxels = torch.zeros(
            size=voxel_params.tensor_size(batch=len(data) * 2),
            device=device,
        )

        targets = torch.zeros(size=(len(data) * 2, 1), device=device)

        for i in range(len(data)):
            rec_pos, rec_neg = data[i]

            rec_pos.voxelize_into(
                voxels, batch_idx=(i * 2), layer_offset=0, cpu=(device.type == 'cpu')
            )
            rec_neg.voxelize_into(
                voxels, batch_idx=(i * 2) + 1, layer_offset=0, cpu=(device.type == 'cpu')
            )

            targets[(i * 2)] = 1
            targets[(i * 2) + 1] = 0

        return (voxels, targets)


if __name__ == "__main__":
    mod = BindingSiteVoxel()
    parser = mod.add_moad_args()
    BindingSiteModel.add_model_args(parser)
    args = parser.parse_args()
    mod.run(args)
