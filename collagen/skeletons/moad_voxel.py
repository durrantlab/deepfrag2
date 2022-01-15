
import argparse
from functools import partial
from multiprocessing import Value
from typing import Type, TypeVar, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
import torch

from ..checkpoints import MyModelCheckpoint, get_last_checkpoint
from .. import VoxelParams, VoxelParamsDefault, MultiLoader
from ..external import MOADInterface


ENTRY_T = TypeVar('ENTRY_T')
TMP_T = TypeVar('TMP_T')
OUT_T = TypeVar('OUT_T')


def _disable_warnings():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    import prody
    prody.confProDy(verbosity="none")


class MoadVoxelSkeleton(object):
    def __init__(self, model_cls: Type[pl.LightningModule], dataset_cls: Type[torch.utils.data.Dataset]):
        self.model_cls = model_cls
        self.dataset_cls = dataset_cls

    @staticmethod
    def build_parser(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # parent_parser = argparse.ArgumentParser()

        parser = parent_parser.add_argument_group('Binding MOAD')

        parser.add_argument("--csv", required=True, help="Path to MOAD every.csv")
        parser.add_argument(
            "--data", required=True, help="Path to MOAD root structure folder"
        )
        parser.add_argument("--cache", required=True, help="Path to MOAD cache.json file")
        parser.add_argument(
            "--split_seed",
            required=False,
            default=1,
            type=int,
            help="Seed for TRAIN/VAL/TEST split. Defaults to 1.",
        )
        parser.add_argument(
            "--num_dataloader_workers",
            default=1,
            type=int,
            help="Number of workers for DataLoader",
        )
        parser.add_argument("--cpu", default=False, action="store_true")
        parser.add_argument("--wandb_project", required=False, default=None)
        parser.add_argument(
            "--max_voxels_in_memory",
            required=True,
            default=80,
            type=int,
            help="The data loader will store no more than this number of voxel in memory at once.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            required=False,
            default=16,
            help="The size of the batch. Defaults to 16.",
        )
        parser.add_argument(
            "-m",
            "--mode",
            type=str,
            choices=['train', 'test'],
            help="Can be train or test. If train, trains the model. If test, runs inference on the test set. Defaults to train.",
            default="train",
        )
        parser.add_argument(
            "--load_checkpoint",
            type=str,
            default=None,
            help="If specified, the model will be loaded from this checkpoint."
        )
        parser.add_argument(
            "--load_newest_checkpoint",
            default=False,
            action='store_true',
            help="If set, the most recent checkpoint will be loaded."
        )
        # TODO: JDD: Load from best validation checkpoint.
        parser.add_argument(
            "--inference_limit",
            default=None,
            help="Maximum number of examples to run inference on."
        )
        parser.add_argument(
            "--inference_rotations",
            default=1,
            type=int,
            help="Number of rotations to sample during inference."
        )

        return parent_parser

    @staticmethod
    def pre_voxelize(args: argparse.Namespace, voxel_params: VoxelParams, entry: ENTRY_T) -> TMP_T:
        return entry

    @staticmethod
    def voxelize(args: argparse.Namespace, voxel_params: VoxelParams, device: torch.device, batch: List[TMP_T]) -> OUT_T:
        raise NotImplementedError()

    @staticmethod
    def batch_eval(args: argparse.Namespace, batch: OUT_T):
        pass

    def _init_trainer(self, args: argparse.Namespace) -> pl.Trainer:
        logger = None
        if args.wandb_project:
            logger = WandbLogger(project=args.wandb_project)
        else:
            logger = CSVLogger(
                "logs", name="my_exp_name", flush_logs_every_n_steps=args.log_every_n_steps
            )

        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=[
                MyModelCheckpoint(
                    dirpath=args.default_root_dir,
                    monitor="val_loss",
                    filename="val-loss-{epoch:02d}-{val_loss:.2f}",
                    save_top_k=3,
                ),
                MyModelCheckpoint(
                    dirpath=args.default_root_dir,
                    monitor="loss",
                    filename="loss-{epoch:02d}-{loss:.2f}",
                    save_last=True,
                    save_top_k=3,
                ),
            ],
        )

        return trainer

    def _init_model(self, args: argparse.Namespace, ckpt: Optional[str]) -> pl.LightningModule:
        if ckpt:
            print(f"Loading model from checkpoint {ckpt}")
            return self.model_cls.load_from_checkpoint(ckpt)
        else:
            return self.model_cls(**vars(args))
    
    def _init_voxel_params(self, args: argparse.Namespace) -> VoxelParams:
        # TODO: make configurable via argparse
        return VoxelParamsDefault.DeepFrag

    def _init_device(self, args: argparse.Namespace) -> torch.device:
        if args.cpu:
            return torch.device('cpu')
        else:
            return torch.device('cuda')

    def _get_checkpoint(self, args: argparse.Namespace) -> Optional[str]:
        if args.load_checkpoint and args.load_newest_checkpoint:
            raise ValueError(f"Can specify 'load_checkpoint=xyz' or 'load_newest_checkpoint' but not both.")
        
        ckpt = None
        if args.load_checkpoint:
            ckpt = args.load_checkpoint
        elif args.load_newest_checkpoint:
            ckpt = get_last_checkpoint()

        return ckpt

    def run(self, args: argparse.Namespace):
        _disable_warnings()
        ckpt = self._get_checkpoint(args)
        if ckpt is not None:
            print(f"Restoring from checkpoint: {ckpt}")

        if args.mode == 'train':
            self._run_train(args, ckpt)
        elif args.mode == 'test':
            self._run_test(args, ckpt)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

    def _run_train(self, args: argparse.Namespace, ckpt: Optional[str]):
        trainer = self._init_trainer(args)
        model = self._init_model(args, ckpt)
        voxel_params = self._init_voxel_params(args)
        device = self._init_device(args)

        moad = MOADInterface(metadata=args.csv, structures=args.data)
        train, val, _ = moad.compute_split(args.split_seed)

        train_dataset = self.dataset_cls(
            moad,
            cache_file=args.cache,
            split=train,
            transform=(lambda entry: self.pre_voxelize(args, voxel_params, entry))
        )
        train_data = (
            MultiLoader(
                train_dataset,
                shuffle=True,
                num_dataloader_workers=args.num_dataloader_workers,
                max_voxels_in_memory=args.max_voxels_in_memory,
            )
            .batch(args.batch_size).map(
                lambda batch: self.voxelize(args, voxel_params, device, batch)
            )
        )

        val_dataset = self.dataset_cls(
            moad,
            cache_file=args.cache,
            split=val,
            transform=(lambda entry: self.pre_voxelize(args, voxel_params, entry))
        )
        val_data = (
            MultiLoader(
                val_dataset,
                shuffle=True,
                num_dataloader_workers=args.num_dataloader_workers,
                max_voxels_in_memory=args.max_voxels_in_memory,
            )
            .batch(args.batch_size).map(
                lambda batch: self.voxelize(args, voxel_params, device, batch)
            )
        )

        trainer.fit(model, train_data, val_data, ckpt_path=ckpt)

    def _run_test(self, args: argparse.Namespace, ckpt: Optional[str]):
        if not ckpt:
            raise ValueError("Must specify a checkpoint in test mode")

        trainer = self._init_trainer(args)
        model = self._init_model(args, ckpt)
        voxel_params = self._init_voxel_params(args)
        device = self._init_device(args)

        moad = MOADInterface(metadata=args.csv, structures=args.data)
        _, _, test = moad.compute_split(args.split_seed)

        test_dataset = self.dataset_cls(
            moad,
            cache_file=args.cache,
            split=test,
            transform=partial(self.__class__.pre_voxelize, args=args, voxel_params=voxel_params)
        )
        test_data = (
            MultiLoader(
                test_dataset,
                shuffle=False,
                num_dataloader_workers=args.num_dataloader_workers,
                # max_voxels_in_memory=args.max_voxels_in_memory,
            )
            .batch(args.batch_size).map(
                partial(self.__class__.voxelize, args=args, voxel_params=voxel_params, device=device)
            )
        )

        model.eval()

        for i in range(args.inference_rotations):
            print(f'Inference rotation {i+1}/{args.inference_rotations}')
            trainer.test(model, test_data, verbose=True)
