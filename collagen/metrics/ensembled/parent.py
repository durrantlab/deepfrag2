"""ParentEnsembled is a parent class for combining multiple predictions
(ensembles of predictions) into one that's hopefully more accurate.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np
import torch
from collagen.core.loader import DataLambda
from collagen.external.moad.types import Entry_info
from collagen.metrics.metrics import PCAProject

# TODO: Only averaged.py uses this. Is the inheritance really necessary?


class ParentEnsembled(ABC):

    """ParentEnsembled is a parent class for combining multiple predictions
    (ensembles of predictions) into one that's hopefully more accurate.
    """

    def __init__(
        self,
        trainer: "pl.Trainer",
        model: "pl.LightningModule",
        test_data: DataLambda,
        num_rotations: int,
        device: torch.device,
        ckpt_name: str,
    ):
        """Initialize the class.

        Args:
            trainer (pl.Trainer): The trainer object.
            model (pl.LightningModule): The model object.
            test_data (DataLambda): The test data.
            num_rotations (int): The number of rotations to perform.
            device (torch.device): The device to use.
            ckpt_name (str): The name of the checkpoint.
        """
        self.device = device
        self.num_rotations = num_rotations
        self.model = model
        self.trainer = trainer
        self.test_data = test_data
        self.ckpt_name = ckpt_name
        self.correct_fp_pca_projected = None
        self.averaged_predicted_fp_pca_projected = None

        # Run it one time to get first-rotation predictions but also the number
        # of entries.
        print(f"{ckpt_name}: Inference rotation 1/{num_rotations}")
        trainer.test(self.model, test_data, verbose=True)
        self.predictions_ensembled = self._create_initial_prediction_tensor()

    def finish(self, pca_space: PCAProject):
        """Finish the ensembling process. Pick up here once you've defined the
        pca_space and label set.
        
        Args:
            pca_space (PCAProject): The PCA space.
        """
        self.pca_space = pca_space

        # Get predictionsPerRotation projection (pca).
        # model.predictions.shape[0] = number of entries
        self.pcas_per_rotation = np.zeros(
            [self.num_rotations, self.model.predictions.shape[0], 2]
        )
        self.pcas_per_rotation[0] = pca_space.project(self.model.predictions)

        # Perform the remaining rotations, adding to predictions_averaged and
        # filling out self.pcas_per_rotation.
        for i in range(1, self.num_rotations):
            print(f"{self.ckpt_name}: Inference rotation {i+1}/{self.num_rotations}")
            self.trainer.test(self.model, self.test_data, verbose=True)
            self.pcas_per_rotation[i] = pca_space.project(self.model.predictions)
            # torch.add(predictions_ensembled, self.model.predictions, out=predictions_ensembled)
            self._udpate_prediction_tensor(self.model.predictions, i)

        self._finalize_prediction_tensor()

    def unpack(self) -> Tuple[Any, torch.Tensor]:
        """Unpack the model and predictions_ensembled.
        
        Returns:
            Tuple[Any, torch.Tensor]: The model and predictions_ensembled.
        """
        return self.model, self.predictions_ensembled

    def get_correct_answer_info(self, entry_idx: int) -> dict:
        """Project correct fingerprints into pca (or other) space.

        Args:
            entry_idx (int): The entry index.

        Returns:
            dict: The correct answer info.
        """
        if self.correct_fp_pca_projected is None:
            self.correct_fp_pca_projected = self.pca_space.project(
                self.model.prediction_targets
            )

        entry_inf: Entry_info = self.model.prediction_targets_entry_infos[entry_idx]
        return {
            "fragmentSmiles": entry_inf.fragment_smiles,
            "pcaProjection": self.correct_fp_pca_projected[entry_idx],
            "parentSmiles": entry_inf.parent_smiles,
            "receptor": entry_inf.receptor_name,
            "connectionPoint": entry_inf.connection_pt.tolist() if entry_inf.connection_pt.size > 0 else None,
        }

    def get_predictions_info(self, entry_idx: int) -> dict:
        """Project averaged predictions into pca (or other) space.

        Args:
            entry_idx (int): The entry index.

        Returns:
            dict: The predictions info.
        """
        if self.averaged_predicted_fp_pca_projected is None:
            self.averaged_predicted_fp_pca_projected = self.pca_space.project(
                self.predictions_ensembled
            )

        return {
            "averagedPrediction": {
                "pcaProjection": self.averaged_predicted_fp_pca_projected[entry_idx],
                "closestFromLabelSet": [],
            },
            "predictionsPerRotation": [
                self.pcas_per_rotation[i][entry_idx].tolist()
                for i in range(self.num_rotations)
            ],
        }

    @abstractmethod
    def _create_initial_prediction_tensor(self):
        """Create the initial prediction tensor. Children should implement
        this.
        """
        pass

    @abstractmethod
    def _udpate_prediction_tensor(self, predicitons_to_add: torch.Tensor, idx: int):
        """Update the prediction tensor. Children should implement this."""
        pass

    @abstractmethod
    def _finalize_prediction_tensor(self):
        """Finalize the prediction tensor. Children should implement this.
        Should modify self.predictions_ensembled directly.
        """
        pass
