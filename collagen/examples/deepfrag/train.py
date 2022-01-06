from typing import List, Tuple
import torch

from collagen import Mol, DelayedMolVoxel, VoxelParams
from collagen.examples.voxel_to_fp_utils.train_utils import (
    FP_SIZE,
    add_args_voxel_to_fp_model,
    train_voxel_to_fp_model,
)
from collagen.external.moad.fragment import MOADFragmentDataset
from collagen.external.moad.moad_interface import MOADInterface
from collagen.util import rand_rot

from collagen.examples.deepfrag.model import DeepFragModel


class PreVoxelize(object):
    """
    Pre-voxelize transform. Given a (receptor, parent, fragment) tuple, prepare
    to voxelize the receptor and parent and compute the fingerprint for the
    fragment.
    """

    def __init__(self, voxel_params: VoxelParams):
        self.voxel_params = voxel_params

    def __call__(
        self, rec: Mol, parent: Mol, frag: Mol
    ) -> Tuple[DelayedMolVoxel, DelayedMolVoxel, torch.Tensor]:
        rot = rand_rot()
        return (
            rec.voxelize_delayed(self.voxel_params, center=frag.connectors[0], rot=rot),
            parent.voxelize_delayed(
                self.voxel_params, center=frag.connectors[0], rot=rot
            ),
            torch.tensor(frag.fingerprint("rdk10", 2048)),
        )


class BatchVoxelize(object):
    """
    Voxelize multiple samples on the GPU.
    """

    def __init__(self, voxel_params: VoxelParams, cpu: bool):
        self.voxel_params = voxel_params
        self.cpu = cpu
        self.device = torch.device("cpu") if cpu else torch.device("cuda")

    def __call__(
        self, data: List[Tuple[DelayedMolVoxel, DelayedMolVoxel, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        voxels = torch.zeros(
            size=self.voxel_params.tensor_size(batch=len(data), feature_mult=2),
            device=self.device,
        )

        fingerprints = torch.zeros(size=(len(data), FP_SIZE), device=self.device)

        for i in range(len(data)):
            rec, parent, frag = data[i]

            rec.voxelize_into(voxels, batch_idx=i, layer_offset=0, cpu=self.cpu)

            parent.voxelize_into(
                voxels,
                batch_idx=i,
                layer_offset=self.voxel_params.atom_featurizer.size(),
                cpu=self.cpu,
            )

            fingerprints[i] = frag

        return (voxels, fingerprints)


def run(args):
    train_voxel_to_fp_model(
        args,
        DeepFragModel,
        MOADInterface,
        MOADFragmentDataset,
        PreVoxelize,
        BatchVoxelize,
    )


if __name__ == "__main__":
    args = add_args_voxel_to_fp_model()
    run(args)
