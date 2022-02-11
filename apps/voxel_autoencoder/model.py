
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class VoxelAutoencoder(pl.LightningModule):
    def __init__(self, latent_size: int = 1024, voxel_features: int = 10):
        super().__init__()
        self.save_hyperparameters()

        # TODO: kwargs not passed?
        # self.learning_rate = kwargs["learning_rate"]

        N = self.hparams.voxel_features
        self.encoder = nn.Sequential(
            nn.Conv3d(N, 32, kernel_size=3, padding=1),
            nn.MaxPool3d(2),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool3d(2),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool3d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3456, self.hparams.latent_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_size, 3456),
            nn.ReLU(),
            nn.Unflatten(1, (128, 3, 3, 3)),
            nn.ConvTranspose3d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                32, N, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )

    @staticmethod
    def add_model_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group('VoxelAutoencoder')
        parser.add_argument('--voxel_features', type=int, help="The number of voxel Features. Defaults to 10.", default=10)
        parser.add_argument('--latent_size', type=int, default=1024)
        return parent_parser

    def forward(self, tensor):
        z = self.encoder(tensor)
        p = self.decoder(z)

        return (z, p)

    def training_step(self, batch, batch_idx):
        _, pred = self(batch)
        loss = F.mse_loss(pred, batch)

        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
