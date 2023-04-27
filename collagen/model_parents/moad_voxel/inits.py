"""Functions to initialize the trainer, model, voxel parameters, and device."""

from typing import Optional
from argparse import Namespace
from collagen.core.voxelization.voxelizer import VoxelParams, VoxelParamsDefault
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from ...checkpoints import MyModelCheckpoint, MyModelCheckpointEveryEpoch
import torch
import os


class MoadVoxelModelInits(object):

    """A few function to initialize the trainer, model, voxel parameters, and
    device.
    """

    @staticmethod
    def init_trainer(args: Namespace) -> pl.Trainer:
        """Initialize the trainer.
        
        Args:
            args: The arguments parsed by argparse.
                
        Returns:
            pl.Trainer: The trainer.
        """
        if args.default_root_dir is not None and not os.path.exists(
            args.default_root_dir
        ):
            os.mkdir(args.default_root_dir)

        if args.wandb_project:
            logger = WandbLogger(project=args.wandb_project)
        elif args.default_root_dir is not None:
            logger = TensorBoardLogger(
                args.default_root_dir + os.sep + "tb_logs", "my_model_run_name"
            )
        else:
            logger = TensorBoardLogger("tb_logs", "my_model_run_name")

        if args.save_every_epoch:
            print("Saving a checkpoint per epoch")
            callbacks = [
                MyModelCheckpointEveryEpoch(
                    dirpath=args.default_root_dir,
                    monitor="val_loss",
                    filename="val-loss-{epoch:02d}-{val_loss:.2f}",
                    save_top_k=args.max_epochs,
                ),
                MyModelCheckpointEveryEpoch(
                    dirpath=args.default_root_dir,
                    monitor="loss",
                    filename="loss-{epoch:02d}-{loss:.2f}",
                    save_last=True,
                    save_top_k=args.max_epochs,
                ),
            ]
        else:
            callbacks = [
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
            ]

        return pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)

    def init_model(
        self: "MoadVoxelModelParent", args: Namespace, ckpt_filename: Optional[str]
    ) -> pl.LightningModule:
        """Initialize the model.

        Args:
            args: The arguments parsed by argparse.
            ckpt_filename: The checkpoint to load from. If None, a new model is
                initialized.

        Returns:
            pl.LightningModule: The model.
        """
        if not ckpt_filename:
            return self.model_cls(**vars(args))

        print(f"\nLoading model from checkpoint {ckpt_filename}\n")
        return self.model_cls.load_from_checkpoint(ckpt_filename)

    @staticmethod
    def init_voxel_params(args: Namespace) -> VoxelParams:
        """Set things like voxel resolution, dimensions, etc. TODO: make
        configurable via argparse. Currently hard coded.

        Args:
            args: The arguments parsed by argparse. 

        Returns:
            VoxelParams: The voxel parameters.
        """
        return VoxelParamsDefault.DeepFrag

    @staticmethod
    def init_device(args: Namespace) -> torch.device:
        """Initialize the device.

        Args:
            args: The arguments parsed by argparse.

        Returns:
            torch.device: The device.
        """
        return torch.device("cpu") if args.cpu else torch.device("cuda")

    def init_warm_model(self, args: Namespace) -> pl.LightningModule:
        """Initialize the model for warm starting (finetuning).

        Args:
            args: The arguments parsed by argparse.

        Returns:
            pl.LightningModule: The model.
        """
        model = self.model_cls(**vars(args))
        state_dict = torch.load(args.model_for_warm_starting)
        model.load_state_dict(state_dict)
        return model
