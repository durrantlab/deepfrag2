from typing import List, Tuple
import numpy as np
import torch

# from scipy.spatial.distance import cdist

from collagen import Mol, DelayedMolVoxel, VoxelParams
from collagen.examples.voxel_to_fp_utils.train_utils import (
    FP_SIZE,
    add_args_voxel_to_fp_model,
    run_voxel_to_fp_model,
)
from collagen.external.moad.moad_interface import MOADInterface
from collagen.external.moad.whole_ligand_to_murcko import MOADMurckoLigDataset as MOADLigDataset
# from collagen.external.moad.whole_ligand import MOADWholeLigDataset as MOADLigDataset
from collagen.util import rand_rot

from collagen.examples.deeplig.model import DeepLigModel


class PreVoxelize(object):
    """
    Pre-voxelize transform. Given a (receptor, ligand) tuple, prepare to
    voxelize the receptor and compute the fingerprint for the ligand.
    """

    def __init__(self, voxel_params: VoxelParams):
        self.voxel_params = voxel_params

    def __call__(self, rec: Mol, ligand: Mol) -> Tuple[DelayedMolVoxel, torch.Tensor]:
        rot = rand_rot()

        passes = False

        # now = time.time()

        # Get the ligand center
        center = np.mean(ligand.coords, axis=0)

        # Add random offset to that.
        center += np.random.uniform(-3, 3, size=(1, 3))[0]
        # print(rec, center)

        # https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html
        # https://github.com/rdkit/rdkit/issues/1947

        # while not passes:
        #     # Get one of the ligand atoms
        #     center=ligand.coords[np.random.randint(ligand.coords.shape[0])]

        #     # Add random offset to that.
        #     center += np.random.uniform(-5,5,size=(1,3))[0]

        #     # Make sure center is not near any receptor atom, nor too far.
        #     min_dist = np.min(cdist(np.array([center]), rec.coords))
        #     if min_dist > 2.0:
        #         if min_dist < 6.0 or time.time() - now > 15:
        #             passes = True

        return (
            rec.voxelize_delayed(self.voxel_params, center=center, rot=rot),
            # parent.voxelize_delayed(
            #     self.voxel_params, center=frag.connectors[0], rot=rot
            # ),
            torch.tensor(ligand.fingerprint("rdk10", 2048)),
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
        self, data: List[Tuple[DelayedMolVoxel, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        voxels = torch.zeros(
            size=self.voxel_params.tensor_size(batch=len(data), feature_mult=2),
            device=self.device,
        )

        fingerprints = torch.zeros(size=(len(data), FP_SIZE), device=self.device)

        for i in range(len(data)):
            rec, lig = data[i]

            rec.voxelize_into(voxels, batch_idx=i, layer_offset=0, cpu=self.cpu)

            # parent.voxelize_into(
            #     voxels,
            #     batch_idx=i,
            #     layer_offset=self.voxel_params.atom_featurizer.size(),
            #     cpu=self.cpu,
            # )

            fingerprints[i] = lig

        return (voxels, fingerprints)


def run(args):
    run_voxel_to_fp_model(
        args,
        DeepLigModel,
        MOADInterface,
        MOADLigDataset,
        PreVoxelize,
        BatchVoxelize,
    )


if __name__ == "__main__":
    args = add_args_voxel_to_fp_model()
    run(args)
