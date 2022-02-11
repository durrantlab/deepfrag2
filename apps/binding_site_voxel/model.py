
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from collagen.metrics import bin_acc


class BindingSiteModel(pl.LightningModule):
    def __init__(self, voxel_features: int = 5, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = kwargs["learning_rate"]

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
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self.loss = nn.BCELoss()

    @staticmethod
    def add_model_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group('BindingSiteModel')
        parser.add_argument('--voxel_features', type=int, help="The number of voxel Features. Defaults to 5.", default=5)
        return parent_parser

    def forward(self, voxel):
        return self.model(voxel)

    def training_step(self, batch, batch_idx):
        if batch is None:
            # Skip invalid batches.
            print("invalid batch")
            return None

        voxel, p = batch
        pred = self(voxel)

        loss = self.loss(pred, p)
        acc = bin_acc(pred, p)

        self.log("loss", loss)
        self.log("acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            # Skip invalid batches.
            return

        voxel, p = batch
        pred = self(voxel)

        loss = self.loss(pred, p)
        acc = bin_acc(pred, p)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
