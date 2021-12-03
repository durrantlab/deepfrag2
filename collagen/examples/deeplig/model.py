import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


_cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def cos(yp, yt):
    """Cosine distance as a loss (inverted)."""
    return 1 - _cos(yp, yt)


class DeepFragModel(pl.LightningModule):
    def __init__(self, voxel_features: int = 10, fp_size: int = 2048):
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

    def forward(self, voxel):
        # self.log("here1", 0)
        return self.model(voxel)

    def training_step(self, batch, batch_idx):
        # self.log("here2", 0)

        voxel, fp = batch
        pred = self(voxel)

        loss = cos(pred, fp).mean()

        # self.print("hi")

        # self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # self.log("here3", 0)
        voxel, fp = batch
        pred = self(voxel)

        loss = cos(pred, fp).mean()

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
