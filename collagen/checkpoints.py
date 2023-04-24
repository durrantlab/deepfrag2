import pytorch_lightning as pl
import glob
import os
import torch


# see  https://github.com/PyTorchLightning/pytorch-lightning/issues/4911 Saves
# and loads checkpoints in a way that respects previously saved checkpoints.
class MyModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
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

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.best_model_score = callback_state["best_model_score"]
        self.best_model_path = callback_state["best_model_path"]
        self.best_model_score = callback_state["best_model_score"]
        self.best_k_models = callback_state["best_k_models"]
        self.save_last = callback_state["save_last"]
        self.kth_best_model_path = callback_state["kth_best_model_path"]


class MyModelCheckpointEveryEpoch(MyModelCheckpoint):
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)

        state_dict_model = trainer._checkpoint_connector.dump_checkpoint(False)["state_dict"]
        torch.save(state_dict_model, filepath + ".pt")


def get_last_checkpoint(args) -> str:
    # Automatically looks for the most recently saved checkpoint. Good for
    # resuming training.

    saved_checkpoints = glob.glob(
        args.default_root_dir + os.sep + "last.ckpt", recursive=True
        # args.default_root_dir + os.sep + "**" + os.sep + "last.ckpt", recursive=True
    )

    if len(saved_checkpoints) == 0:
        raise ValueError("No checkpoints available")

    if len(saved_checkpoints) == 1:
        return saved_checkpoints[0]

    # Multiple saved checkpoints found. Find the most recent one.
    saved_checkpoints = sorted(saved_checkpoints, key=os.path.getmtime, reverse=True)
    return saved_checkpoints[0]
