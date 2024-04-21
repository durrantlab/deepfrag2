import torch
from torch import nn
import torch.nn.functional as F
from apps.deepfrag.model import DeepFragModel
from typing import List, Tuple
from collagen.external.moad.types import Entry_info


class VoxelAutoencoder(DeepFragModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.use_density_net = kwargs["use_density_net"]
        if self.use_density_net:
            self.encoder_decoder = nn.Sequential(
                # Encoder
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Dropout(),
                nn.Linear(8192, self.fp_size),  # Here latent space
                nn.ReLU(),

                # Decoder
                nn.Linear(self.fp_size, 8192),
                nn.ReLU(),
                nn.Unflatten(1, (128, 64)),
                nn.ConvTranspose1d(
                    128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose1d(
                    64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose1d(
                    32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
            )

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:

        if not self.use_density_net:
            return super().forward(voxel)

        df_latent_space = self.encoder(voxel)
        predicted_fps = self.deepfrag_after_encoder(df_latent_space)

        new_latent_space = df_latent_space
        if df_latent_space.size(0) < 16:
            new_latent_space = torch.zeros(16, df_latent_space.size(1))
            new_latent_space[:df_latent_space.size(0), :df_latent_space.size(1)] = df_latent_space

        x = self.encoder_decoder(new_latent_space.unsqueeze(0)).squeeze(0)

        if x.size(0) > df_latent_space.size(0):
            new_x = torch.zeros(df_latent_space.size(0), df_latent_space.size(1))
            new_x[:df_latent_space.size(0), :df_latent_space.size(1)] = x[:df_latent_space.size(0), :df_latent_space.size(1)]
            x = new_x

        return predicted_fps, x, df_latent_space

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, List[Entry_info]], batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, List[Entry_info]]): The
                batch to train on.
            batch_idx (int): The batch index.

        Returns:
            torch.Tensor: The loss.
        """
        if not self.use_density_net:
            return super().training_step(batch, batch_idx)

        voxels, fps, entry_infos = batch

        predicted_fps, x, df_latent_space = self(voxels)

        batch_size = voxels.shape[0]

        pred_loss = self.loss(predicted_fps, fps, entry_infos, batch_size)
        densities_loss = F.mse_loss(x, df_latent_space)
        loss = pred_loss + densities_loss

        self._mark_example_used("train", entry_infos)

        self.log("loss", loss, batch_size=batch_size)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[Entry_info]], batch_idx: int
    ):
        """Run validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, List[Entry_info]]): The
                batch to validate on.
            batch_idx (int): The batch index.
        """
        if not self.use_density_net:
            super().validation_step(batch, batch_idx)
        else:
            voxels, fps, entry_infos = batch

            predicted_fps, x, df_latent_space = self(voxels)

            batch_size = voxels.shape[0]

            pred_loss = self.loss(predicted_fps, fps, entry_infos, batch_size)
            densities_loss = F.mse_loss(x, df_latent_space)
            loss = pred_loss + densities_loss

            self._mark_example_used("val", entry_infos)

            self.log("val_loss", loss, batch_size=batch_size)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[Entry_info]], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Entry_info]]:
        """Run inferance on a given batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, List[Entry_info]]): The
                batch to run inference on.
            batch_idx (int): The batch index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[Entry_info]]: The predicted
                and target fingerprints, and the entry infos.
        """
        if not self.use_density_net:
            return super().test_step(batch, batch_idx)

        voxels, fps, entry_infos = batch

        predicted_fps, x, df_latent_space = self(voxels)

        batch_size = voxels.shape[0]

        pred_loss = self.loss(predicted_fps, fps, entry_infos, batch_size)
        densities_loss = F.mse_loss(x, df_latent_space)
        loss = pred_loss + densities_loss

        self._mark_example_used("test", entry_infos)

        self.log("test_loss", loss, batch_size=batch_size)

        pred = torch.zeros(predicted_fps.size(0), predicted_fps.size(1))
        x = self.deepfrag_after_encoder(x)
        torch.add(input=predicted_fps, other=x, out=pred)
        torch.div(input=pred, other=2, out=pred)

        # Drop (large) voxel input, return the predicted and target fingerprints.
        return pred, fps, entry_infos
