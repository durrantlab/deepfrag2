
import argparse
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
import torch

from collagen import BackedMol, Mol, DelayedMolVoxel, VoxelParams
from collagen.external.moad import MOADInterface
from collagen.external.moad.moad_utils import fix_moad_smiles
from collagen.external.moad import MOADMurckoLigDataset
from collagen.util import rand_rot
from collagen.skeletons import MoadVoxelSkeleton

from model import DeepLigModel


ENTRY_T = Tuple[Mol, Mol] # (rec, ligand)
TMP_T = Tuple[DelayedMolVoxel, torch.Tensor, str, str] # (delayed_voxel, fingerprint, rec_name, lig_name)
OUT_T = Tuple[torch.Tensor, torch.Tensor, List[str], List[str]] # (voxels, fingerprints, rec_names, lig_names)


class DeepLig(MoadVoxelSkeleton):
    def __init__(self):
        super().__init__(
            model_cls=DeepLigModel,
            dataset_cls=MOADMurckoLigDataset,
        )

    @staticmethod
    def pre_voxelize(args: argparse.Namespace, voxel_params: VoxelParams, entry: ENTRY_T) -> TMP_T:
        rec, ligand = entry

        rot = rand_rot()
        rot = np.array([0, 0, 0, 1])

        # passes = False

        # now = time.time()

        # Get the ligand center
        center = np.mean(ligand.coords, axis=0)
        # print(np.mean(rec.coords, axis=0))

        recep_name = rec.meta["name"]
        lig_name = ligand.meta["name"]

        # Add random offset to that.
        # center += np.random.uniform(-3, 3, size=(1, 3))[0]
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

        smi_fixed = fix_moad_smiles(ligand.smiles(True))
        scaffold_smi = MurckoScaffoldSmilesFromSmiles(smi_fixed, includeChirality=True)
        scaffold_rdkit = Chem.MolFromSmiles(scaffold_smi)
        scaffold_mol = BackedMol(scaffold_rdkit, warn_no_confs=False)

        return (
            rec.voxelize_delayed(voxel_params, center=center, rot=rot),
            # parent.voxelize_delayed(
            #     self.voxel_params, center=frag.connectors[0], rot=rot
            # ),
            # TODO: 2048 should be hardcoded here? I think it's a user parameter.
            torch.tensor(scaffold_mol.fingerprint("rdk10", 2048)),
            recep_name, lig_name
        )

    @staticmethod
    def voxelize(args: argparse.Namespace, voxel_params: VoxelParams, device: torch.device, batch: List[TMP_T]) -> OUT_T:
        voxels = torch.zeros(
            size=voxel_params.tensor_size(batch=len(batch), feature_mult=2),
            device=device,
        )

        fingerprints = torch.zeros(size=(len(batch), args.fp_size), device=device)
        rec_names = []
        lig_names = []

        for i in range(len(batch)):
            rec, lig, rec_name, lig_name = batch[i]
            rec_names.append(rec_name)
            lig_names.append(lig_name)

            rec.voxelize_into(voxels, batch_idx=i, layer_offset=0, cpu=(device.type == 'cpu'))

            # parent.voxelize_into(
            #     voxels,
            #     batch_idx=i,
            #     layer_offset=self.voxel_params.atom_featurizer.size(),
            #     cpu=self.cpu,
            # )

            fingerprints[i] = lig

        return (voxels, fingerprints, rec_names, lig_names)


if __name__ == "__main__":
    mod = DeepLig()
    parser = mod.add_moad_args()
    DeepLigModel.add_model_args(parser)
    args = parser.parse_args()
    mod.run(args)
