"""DeepFrag model."""

import argparse
from typing import List, Tuple
from collagen.external.moad.types import Entry_info

from torch import nn
import pytorch_lightning as pl
from apps.deepfrag.AggregationOperators import *
from collagen.metrics import cos_loss, mse_loss


class DeepFragModel(pl.LightningModule):

    """DeepFrag model."""

    def __init__(self, num_voxel_features: int = 10, **kwargs):
        """Initialize the model.

        Args:
            num_voxel_features (int, optional): the number of features per
                voxel. Defaults to 10.
            **kwargs: additional keyword arguments.
        """
        super().__init__()

        self.fp_size = kwargs["fp_size"]
        self.is_regression_mode = kwargs["fragment_representation"] in ["molbert", "normalized_molbert", "shuffled_molbert", "shuffled_normalized_molbert"]
        self.save_hyperparameters()
        self.aggregation = Aggregate1DTensor(operator=kwargs["aggregation_loss_vector"])
        self.learning_rate = kwargs["learning_rate"]
        self.predictions = None
        self.prediction_targets = None
        self.prediction_targets_entry_infos = None

        self._examples_used = {"train": {}, "val": {}, "test": {}}

        self.encoder = nn.Sequential(
            # Rescale data (mean = 0, stdev = 1), per batch.
            nn.BatchNorm3d(num_voxel_features),
            # 3D convolution #1. Output has 64 channels. Each filter is 3x3.
            nn.Conv3d(num_voxel_features, 64, kernel_size=3),
            # Activation function. Output 0 if negative, same if positive.
            nn.ReLU(),
            # 3D convolution #2. Input/output: 64 channels. Each filter is 3x3.
            nn.Conv3d(64, 64, kernel_size=3),
            # Activation function. Output 0 if negative, same if positive.
            nn.ReLU(),
            # 3D convolution #3. Input/output: 64 channels. Each filter is 3x3.
            nn.Conv3d(64, 64, kernel_size=3),
            # Activation function. Output 0 if negative, same if positive.
            nn.ReLU(),
            # Takes max value for each 2x2 field.
            nn.MaxPool3d(kernel_size=2),
            # Rescale data (mean = 0, stdev = 1), per batch.
            nn.BatchNorm3d(64),
            # 3D convolution #4. Input/output: 64 channels. Each filter is 3x3.
            nn.Conv3d(64, 64, kernel_size=3),
            # Activation function. Output 0 if negative, same if positive.
            nn.ReLU(),
            # 3D convolution #5. Input/output: 64 channels. Each filter is 3x3.
            nn.Conv3d(64, 64, kernel_size=3),
            # Activation function. Output 0 if negative, same if positive.
            nn.ReLU(),
            # 3D convolution #6. Input/output: 64 channels. Each filter is 3x3.
            nn.Conv3d(64, 64, kernel_size=3),
            # Activation function. Output 0 if negative, same if positive.
            nn.ReLU(),
            # Calculate the average value of patches to get 1x1x1 output size.
            # nn.AdaptiveAvgPool3d((1, 1, 1)),
            Aggregate3x3Patches(
                operator=kwargs["aggregation_3x3_patches"], output_size=(1, 1, 1)
            ),
            # The dimension of the tensor here is (16, 64, 1, 1, 1)
            # Make the output a vector.
            nn.Flatten(),
            # Randomly zero some values
            nn.Dropout(),
            # Linear transform (fully connected). Increases features to 512.
            nn.Linear(64, 512),
            # Activation function. Output 0 if negative, same if positive.
            nn.ReLU(),
            # Here's your latent space?
        )

        # self.decoder = nn.Sequential(
        #     # Linear transform (fully connected). Increases features to 512.
        #     nn.Linear(512, 64),
        #
        #     # Activation function. Output 0 if negative, same if positive.
        #     nn.ReLU(),
        #
        #     # Reshapes vector to tensor.
        #     nn.Unflatten(1, (64, 1, 1, 1)),
        #
        #     # TODO: Linear layer somewhere here to get it into fragment space?
        #     # Or ReLU?
        #
        #     # Deconvolution #1
        #     nn.ConvTranspose3d(
        #         64, 64, kernel_size=3, stride=1, # padding=1, output_padding=1
        #     ),
        #     nn.ReLU(),
        #
        #     # Deconvolution #2
        #     nn.ConvTranspose3d(
        #         64, 64, kernel_size=3, stride=1, # padding=1, output_padding=1
        #     ),
        #     nn.ReLU(),
        #
        #     # Deconvolution #3
        #     nn.ConvTranspose3d(
        #         64, 64, kernel_size=3, stride=1, # padding=1, output_padding=1
        #     ),
        #     nn.ReLU(),
        #
        #     # Deconvolution #4
        #     nn.ConvTranspose3d(
        #         64, 64, kernel_size=3, stride=1, # padding=1, output_padding=1
        #     ),
        #     nn.ReLU(),
        #
        #     # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #
        #     # Deconvolution #5
        #     nn.ConvTranspose3d(
        #         64, 64, kernel_size=3, stride=1, # padding=1, output_padding=1
        #     ),
        #     nn.ReLU(),
        #
        #     # Deconvolution #6
        #     nn.ConvTranspose3d(
        #         64, 64, kernel_size=3, stride=1, # padding=1, output_padding=1
        #     ),
        #     nn.ReLU(),
        #
        #     # Deconvolution #7
        #     nn.ConvTranspose3d(
        #         64, num_voxel_features, kernel_size=3, stride=1, # padding=1, output_padding=1
        #     ),
        #     nn.ReLU(),
        #
        #     # TODO: num_voxel_features includes receptor + ligand features. Not
        #     # the right one here. Needs to match however you calculate voxel
        #     # fragment.
        # )

        self.deepfrag_after_encoder = nn.Sequential(
            # Randomly zero some values
            nn.Dropout(),
            # Linear transform (fully connected). Increases/Decreases features to the --fp_size argument.
            # It could generate negative values https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            nn.Linear(512, self.fp_size),
        ) if self.is_regression_mode else nn.Sequential(
                # Randomly zero some values
                nn.Dropout(),
                # Linear transform (fully connected). Increases/Decreases features to the --fp_size argument.
                # It could generate negative values https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                nn.Linear(512, self.fp_size),
                # Applies sigmoid activation function. See
                # https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
                # Values ranging between 0 and 1
                nn.Sigmoid(),
            )

    @staticmethod
    def add_model_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add model-specific arguments to the parser.
        
        Args:
            parent_parser (argparse.ArgumentParser): The parser to add to.
            
        Returns:
            argparse.ArgumentParser: The parser with model-specific arguments added.
        """
        # For many of these, good to define default values in args_defaults.py
        parser = parent_parser.add_argument_group("DeepFragModel")
        parser.add_argument(
            "--voxel_features",
            type=int,
            help="The number of voxel Features. Defaults to 10.",
            default=10,
        )
        parser.add_argument(
            "--fragment_representation",
            required=False,
            type=str,
            help="The type of fragment representations to be calculated: rdk10, rdk10_x_morgan, binary_molbert, molbert, normalized_molbert",
        )  # , default="rdk10")
        parser.add_argument(
            "--aggregation_3x3_patches",
            required=False,
            type=str,
            help="The aggregation operator to be used to aggregate 3x3 patches. Defaults to Mean.",
        )  # , default=Operator.MEAN.value)
        parser.add_argument(
            "--aggregation_loss_vector",
            required=False,
            type=str,
            help="The aggregation operator to be used to aggregate loss values. Defaults to Mean.",
        )  # , default=Operator.MEAN.value)
        parser.add_argument(
            "--aggregation_rotations",
            required=False,
            type=str,
            help="The aggregation operator to be used to aggregate rotations. Defaults to Mean.",
        )  # , default=Operator.MEAN.value)
        parser.add_argument(
            "--save_fps",
            action="store_true",
            help="If given, predicted and calculated fingerprints will be saved in binary files during test mode.",
        )
        parser.add_argument(
            "--use_prevalence",
            action="store_true",
            help="If given, prevalence values are calculated and used during fine-tuning on paired data.",
        )
        parser.add_argument(
            "--use_density_net",
            action="store_true",
            help="If given, it is used an additional neural network to predict densities.",
        )
        return parent_parser

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            voxel (torch.Tensor): The voxel grid.
            
        Returns:
            torch.Tensor: The predicted fragment fingerprint.
        """
        latent_space = self.encoder(voxel)
        fps = self.deepfrag_after_encoder(latent_space)
        return fps

    def loss(
        self,
        pred: torch.Tensor,
        fps: torch.Tensor,
        entry_infos: List[Entry_info],
        batch_size: int,
    ) -> torch.Tensor:
        """Calculate the loss.

        Args:
            pred (torch.Tensor): The predicted fragment fingerprint.
            fps (torch.Tensor): The ground truth fragment fingerprint.
            entry_infos (List[Entry_info]): The entry information.
            batch_size (int): The batch size.

        Returns:
            torch.Tensor: The loss.
        """
        return mse_loss(pred, fps) if self.is_regression_mode else self.aggregation.aggregate_on_pytorch_tensor(cos_loss(pred, fps))

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
        voxels, fps, entry_infos = batch

        pred = self(voxels)

        batch_size = voxels.shape[0]

        loss = self.loss(pred, fps, entry_infos, batch_size)

        self._mark_example_used("train", entry_infos)

        self.log("loss", loss, batch_size=batch_size)

        return loss

    def training_epoch_end(self, outputs: List[dict]):
        """Run at the end of the training epoch with the outputs of all
            training steps. Logs the info.
        
        Args:
            outputs (List[dict]): List of outputs you defined in
                training_step(), or if there are multiple dataloaders, a list
                containing a list of outputs for each dataloader.
        """
        # See https://github.com/Lightning-AI/lightning/issues/2110
        try:
            # Sometimes x["loss"] is an empty TensorList. Not sure why. TODO:
            # Using try catch like this is bad practice.
            avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
            self.log(
                "loss_per_epoch", {"avg_loss": avg_loss, "step": self.current_epoch + 1}
            )
        except Exception:
            self.log("loss_per_epoch", {"avg_loss": -1, "step": self.current_epoch + 1})

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[Entry_info]], batch_idx: int
    ):
        """Run validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, List[Entry_info]]): The
                batch to validate on.
            batch_idx (int): The batch index.
        """
        voxels, fps, entry_infos = batch

        pred = self(voxels)

        batch_size = voxels.shape[0]

        loss = self.loss(pred, fps, entry_infos, batch_size)

        self._mark_example_used("val", entry_infos)

        self.log("val_loss", loss, batch_size=batch_size)

    def validation_epoch_end(self, outputs: List[dict]):
        """Run at the end of the validation epoch with the outputs of all
            validation steps. Logs the info.

        Args:
            outputs (List[dict]): List of outputs you defined in
                validation_step(), or if there are multiple dataloaders, a
                list containing a list of outputs for each dataloader.
        """
        # See https://github.com/Lightning-AI/lightning/issues/2110
        try:
            # Sometimes x["val_loss"] is an empty TensorList. Not sure why. TODO:
            # Using try catch like this is bad practice.
            avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            self.log(
                "val_loss_per_epoch",
                {"avg_loss": avg_loss, "step": self.current_epoch + 1},
            )
        except Exception:
            self.log(
                "val_loss_per_epoch", {"avg_loss": -1, "step": self.current_epoch + 1}
            )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        # https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method
        # 3e-4 to 5e-4 are the best learning rates if you're learning the task
        # from scratch.

        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _mark_example_used(self, lbl: str, entry_infos: List[Entry_info]):
        """Mark the example as used.

        Args:
            lbl (str): The label of the split.
            entry_infos (List[Entry_info]): The entry infos.
        """
        if entry_infos is not None:
            for entry_info in entry_infos:
                if entry_info.receptor_name not in self._examples_used[lbl]:
                    # Don't use set here. If one ligand has multiple identical
                    # fragments, I want them all listed.
                    self._examples_used[lbl][entry_info.receptor_name] = []
                self._examples_used[lbl][entry_info.receptor_name].append(
                    entry_info.fragment_smiles
                )

    def get_examples_actually_used(self) -> dict:
        """Get the examples used.

        Returns:
            dict: The examples used.
        """
        to_return = {"counts": {}}
        for split in self._examples_used:
            to_return[split] = {}
            frags_together = []
            for recep in self._examples_used[split].keys():
                frags = self._examples_used[split][recep]
                frags_together.extend(frags)
                to_return[split][recep] = list(frags)
            to_return["counts"][split] = {
                "receptors": len(self._examples_used[split].keys()),
                "fragments": {
                    "total": len(frags_together),
                    "unique": len(set(frags_together)),
                },
            }
        return to_return

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
        voxels, fps, entry_infos = batch
        pred = self(voxels)

        batch_size = voxels.shape[0]

        loss = self.loss(pred, fps, entry_infos, batch_size)

        self._mark_example_used("test", entry_infos)

        self.log("test_loss", loss, batch_size=batch_size)

        # Drop (large) voxel input, return the predicted and target fingerprints.
        return pred, fps, entry_infos

    def test_epoch_end(
        self, results: List[Tuple[torch.Tensor, torch.Tensor, List[Entry_info]]]
    ):
        """Run after inference has been run on all batches.

        Args:
            results (List[Tuple[torch.Tensor, torch.Tensor, List[Entry_info]]]): The
                results from all batches.
        """
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
            key=lambda x: x[0],
        )
        argsort_idx = [i for _, i in keys_and_idxs_sorted]

        prediction_targets_entry_infos = [
            prediction_targets_entry_infos[i] for i in argsort_idx
        ]

        argsort_idx_tnsr = torch.tensor(argsort_idx, device=predictions.device)

        torch.index_select(predictions.clone(), 0, argsort_idx_tnsr, out=predictions)

        torch.index_select(
            prediction_targets.clone(), 0, argsort_idx_tnsr, out=prediction_targets
        )

        # Save predictions, etc., so they can be accessed outside the model.
        self.predictions = predictions
        self.prediction_targets = prediction_targets
        self.prediction_targets_entry_infos = prediction_targets_entry_infos
