"""Functions to initialize the trainer, model, voxel parameters, and device."""

from typing import TYPE_CHECKING, Optional
from argparse import Namespace
from apps.deepfrag.model_paired_data import DeepFragModelPairedDataFinetune
from collagen.core.voxelization.voxelizer import VoxelParams, VoxelParamsDefault
from collagen.external.paired_csv.interface import PairedCsvInterface
import pytorch_lightning as pl  # type: ignore
from pytorch_lightning.loggers.wandb import WandbLogger  # type: ignore
from pytorch_lightning.loggers import TensorBoardLogger  # type: ignore
from ...checkpoints import MyModelCheckpoint, MyModelCheckpointEveryEpoch
import torch  # type: ignore
import os
from collagen.external.common.parent_interface import ParentInterface

if TYPE_CHECKING:
    from collagen.model_parents.moad_voxel.moad_voxel import VoxelModelParent


# A few function to initialize the trainer, model, voxel parameters, and device.
class VoxelModelInits(object):

    """A few function to initialize the trainer, model, voxel parameters, and
    device.
    """

    def __init__(self, parent: "VoxelModelParent"):
        """Initialize the class.

        Args:
            parent (VoxelModelParent): The parent class.
        """
        self.parent = parent

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
        self,
        args: Namespace,
        ckpt_filename: Optional[str],
        fragment_set: Optional["ParentInterface"] = None,
    ) -> pl.LightningModule:
        """Initialize the model.

        Args:
            args: The arguments parsed by argparse.
            ckpt_filename: The checkpoint to load from. If None, a new model is
                initialized.
            fragment_set (optional): the fragment database to be used in test or inference mode

        Returns:
            pl.LightningModule: The model.
        """
        if not ckpt_filename:
            return self.parent.model_cls(**vars(args))

        print(f"\nLoading model from checkpoint {ckpt_filename}\n")
        model = self.parent.model_cls.load_from_checkpoint(ckpt_filename)

        # NOTE: This is how you load the dataset only when using paired data for
        # finetuning.
        if (
            fragment_set
            and isinstance(fragment_set, PairedCsvInterface)
            and isinstance(model, DeepFragModelPairedDataFinetune)
        ):
            model.set_database(fragment_set)
        return model

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

    def init_warm_model(
        self,
        args: Namespace,
        data_interface: ParentInterface,
    ) -> pl.LightningModule:
        """Initialize the model for warm starting (finetuning).

        Args:
            args: The arguments parsed by argparse.
            data_interface: The database (e.g., MOAD) interface.


        Returns:
            pl.LightningModule: The model.
        """
        model = self.parent.model_cls(**vars(args))
        state_dict = torch.load(args.model_for_warm_starting)
        model.load_state_dict(state_dict)
        if isinstance(data_interface, PairedCsvInterface) and isinstance(model, DeepFragModelPairedDataFinetune):
            model.set_database(data_interface)
        return model
