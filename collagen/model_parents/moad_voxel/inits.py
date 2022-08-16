from typing import Any, Type, TypeVar, List, Optional, Tuple, Dict
from argparse import Namespace
from collagen.core.voxelization.voxelizer import VoxelParams, VoxelParamsDefault
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from ...checkpoints import MyModelCheckpoint
import torch

# A few function to initialize the trainer, model, voxel parameters, and device.

class MoadVoxelModelInits(object):
    def init_trainer(self, args: Namespace) -> pl.Trainer:
        logger = None
        if args.wandb_project:
            logger = WandbLogger(project=args.wandb_project)
        else:
            logger = TensorBoardLogger("tb_logs", "my_model_run_name")

        return pl.Trainer.from_argparse_args(
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

    def init_model(
        self: "MoadVoxelModelParent",
        args: Namespace,
        ckpt: Optional[str]
    ) -> pl.LightningModule:
        if ckpt:
            print(f"\nLoading model from checkpoint {ckpt}\n")
            return self.model_cls.load_from_checkpoint(ckpt)
        else:
            return self.model_cls(**vars(args))

    def init_voxel_params(self, args: Namespace) -> VoxelParams:
        # Sets things like voxel resolution, dimensions, etc. TODO: make
        # configurable via argparse. Currently hard coded.
        return VoxelParamsDefault.DeepFrag

    def init_device(self, args: Namespace) -> torch.device:
        return torch.device("cpu") if args.cpu else torch.device("cuda")
