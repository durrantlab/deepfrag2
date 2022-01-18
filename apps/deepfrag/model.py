import argparse

import torch
from torch import nn
import pytorch_lightning as pl

from collagen.metrics import cos_loss, top_k


class DeepFragModel(pl.LightningModule):
    def __init__(self, voxel_features: int = 10, fp_size: int = 2048, **kwargs):
        super().__init__()
        self.save_hyperparameters()

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
        voxel, fp = batch
        pred = self(voxel)

        loss = cos_loss(pred, fp).mean()

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        voxel, fp = batch
        pred = self(voxel)

        loss = cos_loss(pred, fp).mean()

        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        voxel, fp = batch
        pred = self(voxel)

        loss = cos_loss(pred, fp).mean()
        self.log("test_loss", loss)

        # Drop (large) voxel input, return the predicted and target fingerprints.
        return pred, fp

    def test_epoch_end(self, results):
        predictions = torch.cat([x[0] for x in results])
        targets = torch.cat([x[1] for x in results])

        fp = targets.unique(dim=0)

        self.log('LBL_TEST_SIZE', len(fp))

        top = top_k(predictions, targets, fp, k=[1,8,16,32,64])

        for k in top:
            self.log(f'test_top_{k}', top[k])
