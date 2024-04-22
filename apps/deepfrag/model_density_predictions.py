import torch
from torch import nn
from apps.deepfrag.model import DeepFragModel
from typing import List, Tuple
from collagen.external.moad.types import Entry_info


class VoxelAutoencoder(DeepFragModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.use_density_net = kwargs["use_density_net"]
        if self.use_density_net:
            self.density_encoder = nn.Sequential(
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
                nn.Linear(8192, 512),  # Here latent space
                nn.ReLU(),
            )
            self.density_decoder = nn.Sequential(
                # Decoder
                nn.Linear(512, 8192),
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
        """Forward pass of the model.

        Args:
            voxel (torch.Tensor): The voxel grid.

        Returns:
            torch.Tensor: The predicted fragment fingerprint, the recovered DF latent space, the DF latent space, and the density latent spce
        """
        if not self.use_density_net:
            return super().forward(voxel)

        df_latent_space = self.encoder(voxel)

        aux_df_latent_space = df_latent_space
        if df_latent_space.size(0) < 16:
            aux_df_latent_space = torch.zeros(16, df_latent_space.size(1))
            aux_df_latent_space[:df_latent_space.size(0), :df_latent_space.size(1)] = df_latent_space

        den_latent_space = self.density_encoder(aux_df_latent_space.unsqueeze(0))
        pred_df_latent_space = self.density_decoder(den_latent_space).squeeze(0)

        if pred_df_latent_space.size(0) > df_latent_space.size(0):
            aux = torch.zeros(df_latent_space.size(0), df_latent_space.size(1))
            aux[:df_latent_space.size(0), :df_latent_space.size(1)] = pred_df_latent_space[:df_latent_space.size(0), :df_latent_space.size(1)]
            pred_df_latent_space = aux

        return self.deepfrag_after_encoder(df_latent_space), pred_df_latent_space, df_latent_space, den_latent_space

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

        predicted_fps, pred_df_latent_space, df_latent_space, _ = self(voxels)

        batch_size = voxels.shape[0]

        pred_loss = self.loss(predicted_fps, fps, entry_infos, batch_size)
        densities_loss = nn.MSELoss(reduction='sum')(pred_df_latent_space, df_latent_space)
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

            predicted_fps, pred_df_latent_space, df_latent_space, _ = self(voxels)

            batch_size = voxels.shape[0]

            pred_loss = self.loss(predicted_fps, fps, entry_infos, batch_size)
            densities_loss = nn.MSELoss(reduction='sum')(pred_df_latent_space, df_latent_space)
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

        predicted_fps, pred_df_latent_space, df_latent_space, den_latent_space = self(voxels)

        batch_size = voxels.shape[0]

        pred_loss = self.loss(predicted_fps, fps, entry_infos, batch_size)
        densities_loss = nn.MSELoss(reduction='sum')(pred_df_latent_space, df_latent_space)
        loss = pred_loss + densities_loss

        self._mark_example_used("test", entry_infos)

        self.log("test_loss", loss, batch_size=batch_size)

        pred = torch.zeros(predicted_fps.size(0), predicted_fps.size(1))
        torch.add(input=predicted_fps, other=self.deepfrag_after_encoder(den_latent_space), out=pred)
        torch.div(input=pred, other=2, out=pred)

        # Drop (large) voxel input, return the predicted and target fingerprints.
        return pred, fps, entry_infos
