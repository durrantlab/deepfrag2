import argparse

import torch
from torch import nn
import pytorch_lightning as pl

from collagen.metrics import cos_loss


class DeepFragModel(pl.LightningModule):
    def __init__(self, voxel_features: int = 10, fp_size: int = 2048, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = kwargs["learning_rate"]

        self.examples_used = {
            "train": {},
            "val": {},
            "test": {}
        }

        self.model = nn.Sequential(
            nn.BatchNorm3d(voxel_features),
            nn.Conv3d(voxel_features, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, fp_size),
            nn.Sigmoid(),
        )

    @staticmethod
    def add_model_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group('DeepFragModel')
        parser.add_argument('--voxel_features', type=int, help="The number of voxel Features. Defaults to 10.", default=10)
        parser.add_argument('--fp_size', type=int, help="The size of the output molecular fingerprint. Defaults to 2048.", default=2048)
        return parent_parser

    def forward(self, voxel):
        return self.model(voxel)

    def training_step(self, batch, batch_idx):
        voxels, fps, entry_infos = batch

        pred = self(voxels)

        loss = cos_loss(pred, fps).mean()

        print(entry_infos.fragment_smiles, entry_infos.receptor_name)

        # print("training_step")
        self.log("loss", loss, batch_size=voxels.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        voxels, fps, entry_infos = batch

        # print("::", voxels.shape, fps.shape, len(smis))

        pred = self(voxels)

        loss = cos_loss(pred, fps).mean()

        # print("validation_step")
        self.log("val_loss", loss, batch_size=voxels.shape[0])

    def configure_optimizers(self):
        # https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method
        # 3e-4 to 5e-4 are the best learning rates if you're learning the task
        # from scratch
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def test_step(self, batch, batch_idx):
        # Runs inferance on a given batch.
        voxels, fps, entry_infos = batch
        pred = self(voxels)

        loss = cos_loss(pred, fps).mean()

        # print("test_step")
        self.log("test_loss", loss, batch_size=voxels.shape[0])

        # Drop (large) voxel input, return the predicted and target fingerprints.
        return pred, fps, entry_infos

    def test_epoch_end(self, results):
        # This runs after inference has been run on all batches.

        predictions = torch.cat([x[0] for x in results])
        prediction_targets = torch.cat([x[1] for x in results])

        prediction_targets_entry_infos = []
        for x in results:
            prediction_targets_entry_infos.extend(x[2])

        # Sort so that order is always the same (for multiple rotations).
        keys_and_idxs_sorted = sorted(
            [
                (e.hashable_key(), i) 
                for i, e in enumerate(prediction_targets_entry_infos)
            ], 
            key=lambda x: x[0]
        )
        argsort_idx = [i for _, i in keys_and_idxs_sorted]

        prediction_targets_entry_infos = [
            prediction_targets_entry_infos[i] for i in argsort_idx
        ]

        argsort_idx_tnsr = torch.tensor(argsort_idx, device=predictions.device)

        torch.index_select(
            predictions.clone(), 0, argsort_idx_tnsr,
            out=predictions
        )

        torch.index_select(
            prediction_targets.clone(), 0, argsort_idx_tnsr,
            out=prediction_targets
        )

        # Save predictions, etc., so they can be accessed outside the model.
        self.predictions = predictions
        self.prediction_targets = prediction_targets
        self.prediction_targets_entry_infos = prediction_targets_entry_infos

