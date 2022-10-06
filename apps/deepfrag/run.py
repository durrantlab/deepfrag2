import argparse
import torch
import pytorch_lightning as pl

from typing import List, Tuple
from collagen.external.moad.types import Entry_info
from collagen import Mol, DelayedMolVoxel, VoxelParams
from collagen.external.moad import MOADFragmentDataset
from collagen.util import rand_rot
from collagen.model_parents import MoadVoxelModelParent
from collagen.core.args import get_args
from apps.deepfrag.model import DeepFragModel
from apps.deepfrag.AggregationOperators import Operator

ENTRY_T = Tuple[Mol, Mol, Mol]
TMP_T = Tuple[DelayedMolVoxel, DelayedMolVoxel, torch.Tensor, str]
OUT_T = Tuple[torch.Tensor, torch.Tensor, List[str]]


def _fingerprint_fn(args: argparse.Namespace, mol: Mol):
    return torch.tensor(mol.fingerprint("rdk10", args.fp_size))


class DeepFrag(MoadVoxelModelParent):
    def __init__(self):
        super().__init__(
            model_cls=DeepFragModel, dataset_cls=MOADFragmentDataset
        )

    @staticmethod
    def pre_voxelize(
        args: argparse.Namespace, voxel_params: VoxelParams, entry: ENTRY_T
    ) -> TMP_T:
        rec, parent, frag = entry
        rot = rand_rot()
        center = frag.connectors[0]

        payload = Entry_info(
            fragment_smiles=frag.smiles(True),
            parent_smiles=parent.smiles(True),
            receptor_name=rec.meta["name"],
            connection_pt=center
        )

        # if rec.meta["name"] == "Receptor 2v0u":
        #     print(["2", rec.meta["name"], frag.smiles()])

        return (
            rec.voxelize_delayed(voxel_params, center=center, rot=rot),
            parent.voxelize_delayed(voxel_params, center=center, rot=rot),
            _fingerprint_fn(args, frag),
            payload
        )

    @staticmethod
    def voxelize(
        args: argparse.Namespace,
        voxel_params: VoxelParams,
        device: torch.device,
        batch: List[TMP_T]
    ) -> OUT_T:

        voxels = torch.zeros(
            size=voxel_params.tensor_size(batch=len(batch), feature_mult=2),
            device=device,
        ) if voxel_params.calc_voxels else None

        fingerprints = torch.zeros(
            size=(len(batch), args.fp_size), device=device
        ) if voxel_params.calc_fps else None
        
        frag_smis = []

        for i in range(len(batch)):
            rec, parent, frag, smi = batch[i]

            if voxel_params.calc_voxels:
                rec.voxelize_into(
                    voxels, batch_idx=i, layer_offset=0, cpu=(device.type == "cpu")
                )

                parent.voxelize_into(
                    voxels,
                    batch_idx=i,
                    layer_offset=voxel_params.atom_featurizer.size(),
                    cpu=(device.type == "cpu"),
                )

            if voxel_params.calc_fps:
                fingerprints[i] = frag
            
            frag_smis.append(smi)

        return voxels, fingerprints, frag_smis


def function_2run_deepfrag():
    print("PyTorch", torch.__version__)
    print("PytorchLightning", pl.__version__)

    args = get_args(
        parser_funcs=[
            MoadVoxelModelParent.add_moad_args, DeepFragModel.add_model_args, 
            MOADFragmentDataset.add_fragment_args
        ],
        post_parse_args_funcs=[MoadVoxelModelParent.fix_moad_args],
        is_pytorch_lightning=True,
    )
    # args.__setattr__("mode", "test")
    # args.__setattr__("load_newest_checkpoint", True)
    # args.__setattr__("inference_rotations", 8)

    # args.__setattr__("cpu", True)
    # args.__setattr__("save_splits", "D:\\Cesar\\0.Investigacion\\3.Experimentacion\\DeepFrag\\Datasets\\splits.json")
    # args.__setattr__("default_root_dir", "D:\\Cesar\\0.Investigacion\\3.Experimentacion\\DeepFrag\\Datasets")

    args.__setattr__("accelerator", "gpu")
    args.__setattr__("devices", 1)
    args.__setattr__("save_splits", "/home/crg93/Data/crg93/output_deepfrag2/splits.json")
    args.__setattr__("default_root_dir", "/home/crg93/Data/crg93/output_deepfrag2/")

    args.__setattr__("verbose", True)
    args.__setattr__("max_epochs", 30)
    args.__setattr__("aggregation_3x3_patches", Operator.MEAN)
    args.__setattr__("aggregation_loss_vector", Operator.MEAN)
    args.__setattr__("aggregation_rotations", Operator.MEAN)
    args.__setattr__("inference_rotations", 8)
    # args.__setattr__("cache_pdbs_to_disk", True)
    args.__setattr__("csv", "/mnt/Data/crg93/moad.updated/every.csv")
    args.__setattr__("data", "/mnt/Data/crg93/moad.updated/BindingMOAD_2020/")
    args.__setattr__("cache", "/mnt/Data/crg93/moad.updated/every.csv.cache.json")

    import pdb; pdb.set_trace()

    model = DeepFrag()
    model.run(args)


if __name__ == "__main__":
    function_2run_deepfrag()
