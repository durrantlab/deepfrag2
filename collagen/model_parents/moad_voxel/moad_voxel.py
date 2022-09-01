from argparse import ArgumentParser, Namespace
import json
from typing import Type, TypeVar, List
from collagen.model_parents.moad_voxel.inits import MoadVoxelModelInits
from collagen.model_parents.moad_voxel.lr_finder import MoadVoxelModelLRFinder
from collagen.model_parents.moad_voxel.test import MoadVoxelModelTest
from collagen.model_parents.moad_voxel.train import MoadVoxelModelTrain
from collagen.model_parents.moad_voxel import arguments
import os
from collagen.model_parents.moad_voxel.utils import MoadVoxelModelUtils

import pytorch_lightning as pl
import torch

from ... import VoxelParams

ENTRY_T = TypeVar("ENTRY_T")
TMP_T = TypeVar("TMP_T")
OUT_T = TypeVar("OUT_T")


class MoadVoxelModelParent(
    MoadVoxelModelInits,
    MoadVoxelModelLRFinder,
    MoadVoxelModelTest,
    MoadVoxelModelTrain,
    MoadVoxelModelUtils,
):
    NUM_MOST_SIMILAR_PER_ENTRY = 5

    def __init__(
        self,
        model_cls: Type[pl.LightningModule],
        dataset_cls: Type[torch.utils.data.Dataset],
    ):
        self.model_cls = model_cls
        self.dataset_cls = dataset_cls

    @staticmethod
    def add_moad_args(parent_parser: ArgumentParser) -> ArgumentParser:
        return arguments.add_moad_args(parent_parser)

    @staticmethod
    def fix_moad_args(args: Namespace) -> Namespace:
        # Only works after arguments have been parsed, so in a separate
        # definition.
        return arguments.fix_moad_args(args)

    @staticmethod
    def pre_voxelize(
        args: Namespace, voxel_params: VoxelParams, entry: ENTRY_T
    ) -> TMP_T:
        # Should be overwritten by child class.
        return entry

    @staticmethod
    def voxelize(
        args: Namespace,
        voxel_params: VoxelParams,
        device: torch.device,
        batch: List[TMP_T],
    ) -> OUT_T:
        # Should be overwritten by child class.
        raise NotImplementedError()

    @staticmethod
    def batch_eval(args: Namespace, batch: OUT_T):
        pass

    @staticmethod
    def custom_test(args: Namespace, predictions):
        pass

    def run(self, args: Namespace = None):
        self.disable_warnings()

        ckpt = self.get_checkpoint(args)
        if ckpt is not None:
            print(f"Restoring from checkpoint: {ckpt}")

        if args.mode == "train":
            self.run_train(args, ckpt)
        elif args.mode == "test":
            self.run_test(args, ckpt)
        elif args.mode == "lr_finder":
            self.run_lr_finder(args)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

    # def _get_train_frag_counts(self, args, moad, train, voxel_params, device):
    #     # Without calculating voxels and fingerprints...

    #     # TODO: No longer used, but leaving here as an example of how to get
    #     # fragments without calculating voxels. Useful for testing?

    #     voxel_params_frag = copy.deepcopy(voxel_params)
    #     voxel_params_frag.calc_voxels = False
    #     voxel_params_frag.calc_fps = False
    #     train_data_to_get_frags = get_data_from_split(self,
    #         args, moad, train, voxel_params_frag, device
    #     )

    #     frag_counts = {}

    #     for payload in tqdm(train_data_to_get_frags, desc="Counting fragment SMILES..."):
    #         _, _, entry_infos = payload
    #         for entry_info in entry_infos:
    #             if entry_info.fragment_smiles not in frag_counts:
    #                 # Don't use set here. If one ligand has multiple identical
    #                 # fragments, I want them all listed.
    #                 frag_counts[entry_info.fragment_smiles] = 0
    #             frag_counts[entry_info.fragment_smiles] += 1
    #     return frag_counts

    @staticmethod
    def _save_examples_used(model, args):
        if args.save_splits is not None:
            pth = "/mnt/extra/" if os.path.exists("/mnt/extra/") else ""
            examples_used = model.get_examples_used()
            out_name = pth + os.path.basename(args.save_splits) + ".actually_used.json"
            json.dump(examples_used, open(out_name, "w"), indent=4)
