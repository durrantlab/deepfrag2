"""Run DeepFrag."""

import argparse
from apps.deepfrag.model_paired_data import DeepFragModelPairedDataFinetune
from collagen.core.molecules.mol import BackedMol
import torch
import pytorch_lightning as pl

from typing import List, Tuple, Union
from collagen.external.moad.types import Entry_info
from collagen import Mol, DelayedMolVoxel, VoxelParams
from collagen.external.moad import MOADFragmentDataset
from collagen.util import rand_rot
from collagen.model_parents import MoadVoxelModelParent
from collagen.core.args import get_args
from apps.deepfrag.model import DeepFragModel
from apps.deepfrag.model_density_predictions import VoxelAutoencoder

ENTRY_T = Tuple[Mol, Mol, Mol]
TMP_T = Tuple[DelayedMolVoxel, DelayedMolVoxel, torch.Tensor, str]
OUT_T = Tuple[torch.Tensor, torch.Tensor, List[str]]


def _fingerprint_fn(args: argparse.Namespace, mol: BackedMol):
    return torch.tensor(mol.fingerprint(args.fragment_representation, args.fp_size))


class DeepFrag(MoadVoxelModelParent):

    """DeepFrag model."""

    def __init__(self, args: argparse.Namespace):
        """Initialize the DeepFrag model parent.
        
        Args:
            args (argparse.Namespace): The arguments.
        """
        super().__init__(
            model_cls=DeepFragModelPairedDataFinetune
            if args.paired_data_csv
            else VoxelAutoencoder
            if args.use_density_net
            else DeepFragModel,
            dataset_cls=MOADFragmentDataset,
        )

    @staticmethod
    def pre_voxelize(
        args: argparse.Namespace, voxel_params: VoxelParams, entry: ENTRY_T
    ) -> TMP_T:
        """Preprocess the entry before voxelization.
        
        Args:
            args (argparse.Namespace): The arguments parsed by argparse.
            voxel_params (VoxelParams): The voxelization parameters.
            entry (ENTRY_T): The entry to preprocess.
            
        Returns:
            TMP_T: The preprocessed entry.
        """
        rec, parent, frag, ligand_id, fragment_idx = entry
        rot = rand_rot()
        center = frag.connectors[0]

        payload = Entry_info(
            fragment_smiles=frag.smiles(True),
            parent_smiles=parent.smiles(True),
            receptor_name=rec.meta["name"],
            connection_pt=center,
            ligand_id=ligand_id,
            fragment_idx=fragment_idx,
        )

        # if rec.meta["name"] == "Receptor 2v0u":
        #     print(["2", rec.meta["name"], frag.smiles()])

        return (
            rec.voxelize_delayed(voxel_params, center=center, rot=rot),
            parent.voxelize_delayed(voxel_params, center=center, rot=rot),
            _fingerprint_fn(args, frag),
            payload,
        )

    @staticmethod
    def voxelize(
        args: argparse.Namespace,
        voxel_params: VoxelParams,
        device: torch.device,
        batch: List[TMP_T],
    ) -> OUT_T:
        """Voxelize the batch.
        
        Args:
            args (argparse.Namespace): The arguments parsed by argparse.
            voxel_params (VoxelParams): The voxelization parameters.
            device (torch.device): The device to use.
            batch (List[TMP_T]): The batch to voxelize.
            
        Returns:
            OUT_T: The voxels, fingerprints, and fragment SMILES.
        """
        voxels = (
            torch.zeros(
                size=voxel_params.tensor_size(batch=len(batch), feature_mult=2),
                device=device,
            )
            if voxel_params.calc_voxels
            else None
        )

        fingerprints: Union[torch.Tensor, None] = (
            torch.zeros(size=(len(batch), args.fp_size), device=device)
            if voxel_params.calc_fps
            else None
        )

        frag_smis = []

        for i in range(len(batch)):
            rec, parent, frag, smi = batch[i]

            if voxel_params.calc_voxels:
                rec.voxelize_into(
                    voxels, batch_idx=i, layer_offset=0, cpu=(device.type == "cpu")
                )

                # atom_featurizer must not be None
                assert (
                    voxel_params.atom_featurizer is not None
                ), "Atom featurizer is None"

                parent.voxelize_into(
                    voxels,
                    batch_idx=i,
                    layer_offset=voxel_params.atom_featurizer.size(),
                    cpu=(device.type == "cpu"),
                )

            if voxel_params.calc_fps:
                # Make sure fingerprint is not None
                assert fingerprints is not None, "Fingerprint tensor is None"
                fingerprints[i] = frag

            frag_smis.append(smi)

        return voxels, fingerprints, frag_smis


def function_to_run_deepfrag():
    """Run DeepFrag."""
    print("PyTorch", torch.__version__)
    print("PytorchLightning", pl.__version__)

    args = get_args(
        parser_funcs=[
            MoadVoxelModelParent.add_moad_args,
            DeepFragModel.add_model_args,
            MOADFragmentDataset.add_fragment_args,
        ],
        post_parse_args_funcs=[MoadVoxelModelParent.fix_moad_args],
        is_pytorch_lightning=True,
    )

    model = DeepFrag(args)
    model.run(args)
