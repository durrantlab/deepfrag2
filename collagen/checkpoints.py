"""Contains classes and functions for saving and loading checkpoints."""

import pytorch_lightning as pl  # type: ignore
import glob
import os
import torch  # type: ignore
import argparse

# see  https://github.com/PyTorchLightning/pytorch-lightning/issues/4911 Saves
# and loads checkpoints in a way that respects previously saved checkpoints.


class MyModelCheckpoint(pl.callbacks.ModelCheckpoint):

    """Save a checkpoint when the monitored metric improves. Inherits from the
    ModelCheckpoint class.
    """

    @property
    def state_key(self) -> str:
        """Generate a unique state key for this callback instance.
        
        This is required when multiple ModelCheckpoint callbacks are used
        in the same Trainer to avoid state collision errors in PL >= 1.7.
        """
        # We use the class name, monitor, and filename template to ensure uniqueness
        # since inits.py uses distinct values for these across the 3 instances.
        return f"{self.__class__.__name__}_{self.monitor}_{self.filename}"

    def state_dict(self) -> dict:
        """Save the callback state.
        
        This replaces on_save_checkpoint from earlier versions. It is the
        standard PyTorch method for state persistence and works across virtually
        all PL versions.
        """
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
            "dirpath": self.dirpath,
            "best_k_models": self.best_k_models,
            "save_last": self.save_last,
            "kth_best_model_path": self.kth_best_model_path,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the callback state.
        
        This replaces on_load_checkpoint from previous versions. It avoids the
        signature mismatches introduced in PL 1.8.
        """
        self.best_model_score = state_dict["best_model_score"]
        self.best_model_path = state_dict["best_model_path"]
        self.best_k_models = state_dict["best_k_models"]
        self.save_last = state_dict["save_last"]
        self.kth_best_model_path = state_dict["kth_best_model_path"]


class MyModelCheckpointEveryEpoch(MyModelCheckpoint):

    """Save a checkpoint at the end of every epoch. Inherits from the
    ModelCheckpoint class.
    """

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Save a checkpoint.

        Args:
            trainer (pl.Trainer): The trainer object.
            filepath (str): The path to save the checkpoint.
        """
        super()._save_checkpoint(trainer, filepath)
        # Note: This relies on internal implementation details of ModelCheckpoint.
        # dump_checkpoint returns the checkpoint dictionary.
        state_dict_model = trainer._checkpoint_connector.dump_checkpoint(False)[
            "state_dict"
        ]
        torch.save(state_dict_model, f"{filepath}.pt")


def get_last_checkpoint(args: "argparse.Namespace") -> str:
    """Automatically looks for the most recently saved checkpoint. Good for
    resuming training.

    Args:
        args (argparse.Namespace): The arguments.

    Raises:
        ValueError: If no checkpoints are available.

    Returns:
        str: The path to the last saved checkpoint.
    """
    saved_checkpoints = glob.glob(
        args.default_root_dir + os.sep + "last.ckpt",
        recursive=True
        # args.default_root_dir + os.sep + "**" + os.sep + "last.ckpt", recursive=True
    )

    if not saved_checkpoints:
        raise ValueError("No checkpoints available")

    if len(saved_checkpoints) == 1:
        return saved_checkpoints[0]

    # Multiple saved checkpoints found. Find the most recent one.
    saved_checkpoints = sorted(saved_checkpoints, key=os.path.getmtime, reverse=True)
    return saved_checkpoints[0]
