
import argparse

import torch
from torch import nn
import pytorch_lightning as pl

from collagen.metrics import cos_loss


class DeepLigModel(pl.LightningModule):
    def __init__(self, voxel_features: int = 10, fp_size: int = 2048, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = 1e-3

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
        parser = parent_parser.add_argument_group('DeepLigModel')
        parser.add_argument('--voxel_features', type=int, help="The number of voxel Features. Defaults to 10.", default=10)
        parser.add_argument('--fp_size', type=int, help="The size of the output molecular fingerprint. Defaults to 2048.", default=2048)
        return parent_parser

    def forward(self, voxel):
        return self.model(voxel)

    def training_step(self, batch, batch_idx):
        voxel, fp, recs, ligs = batch
        pred = self(voxel)

        loss = cos_loss(pred, fp).mean()

        self.log("loss", loss)
        # self.log("learning_rate", self.learning_rate)

        # For debugging...
        # num_file_descriptors = int(subprocess.check_output("lsof | wc -l", shell=True).strip())
        # self.log("num_files", num_file_descriptors)

        return loss

    def validation_step(self, batch, batch_idx):
        voxel, fp, recs, ligs = batch
        pred = self(voxel)

        loss = cos_loss(pred, fp).mean()

        self.log("val_loss", loss)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # this calls forward
        return self(batch)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # print(self.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
