"""Contains classes and functions for saving and loading checkpoints."""

import pytorch_lightning as pl
import glob
import os
import torch
import argparse

# see  https://github.com/PyTorchLightning/pytorch-lightning/issues/4911 Saves
# and loads checkpoints in a way that respects previously saved checkpoints.


class MyModelCheckpoint(pl.callbacks.ModelCheckpoint):

    """Save a checkpoint when the monitored metric improves. Inherits from the
    ModelCheckpoint class.
    """

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module, checkpoint) -> dict:
        """Run when saving a checkpoint.

        Args:
            trainer (pl.Trainer): Not used...
            pl_module (any): Not used...
            checkpoint (any): Not used...

        Returns:
            dict: The dictionary of values to save.
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

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module, callback_state):
        """Run when loading a checkpoint.

        Args:
            trainer (pl.Trainer): Not used...
            pl_module (any): Not used...
            callback_state (dict): The state of values loaded.
        """
        self.best_model_score = callback_state["best_model_score"]
        self.best_model_path = callback_state["best_model_path"]
        self.best_model_score = callback_state["best_model_score"]
        self.best_k_models = callback_state["best_k_models"]
        self.save_last = callback_state["save_last"]
        self.kth_best_model_path = callback_state["kth_best_model_path"]


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
