from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch
from collagen.core.loader import DataLambda
from collagen.external.moad.types import Entry_info
from collagen.metrics.metrics import VisRepProject

class ParentEnsembled(ABC):
    def __init__(self, trainer: Any, model: Any, test_data: DataLambda, num_rotations: int, device: Any, ckpt_name: str):
        self.device = device
        self.num_rotations = num_rotations
        self.model = model
        self.trainer = trainer
        self.test_data = test_data
        self.ckpt_name = ckpt_name
        self.correct_fp_vis_rep_projected = None
        self.averaged_predicted_fp_vis_rep_projected = None
        
        # Run it one time to get first-rotation predictions but also the number
        # of entries.
        print(f"{ckpt_name}: Inference rotation 1/{num_rotations}")
        trainer.test(self.model, test_data, verbose=True)
        self.predictions_ensembled = self._create_initial_prediction_tensor()

    def finish(self, vis_rep_space: VisRepProject):
        # Pick up here once you've defined the vis_rep_space and label set.

        self.vis_rep_space = vis_rep_space

        # Get predictionsPerRotation projection (pca).
        # model.predictions.shape[0] = number of entries
        self.viz_reps_per_rotation = np.zeros([self.num_rotations, self.model.predictions.shape[0], 2])
        self.viz_reps_per_rotation[0] = vis_rep_space.project(self.model.predictions)

        # Perform the remaining rotations, adding to predictions_averaged and
        # filling out self.viz_reps_per_rotation.
        for i in range(1, self.num_rotations):
            print(f"{self.ckpt_name}: Inference rotation {i+1}/{self.num_rotations}")
            self.trainer.test(self.model, self.test_data, verbose=True)
            self.viz_reps_per_rotation[i] = vis_rep_space.project(self.model.predictions)
            # torch.add(predictions_ensembled, self.model.predictions, out=predictions_ensembled)
            self._udpate_prediction_tensor(self.model.predictions, i)

        self._finalize_prediction_tensor()

    def unpack(self):
        return self.model, self.predictions_ensembled

    def get_correct_answer_info(self, entry_idx: int):
        # Project correct fingerprints into pca (or other) space.
        if self.correct_fp_vis_rep_projected is None:
            self.correct_fp_vis_rep_projected = self.vis_rep_space.project(self.model.prediction_targets)

        entry_inf: Entry_info = self.model.prediction_targets_entry_infos[entry_idx]
        return {
            "fragmentSmiles": entry_inf.fragment_smiles,
            "vizRepProjection": self.correct_fp_vis_rep_projected[entry_idx],
            "parentSmiles": entry_inf.parent_smiles,
            "receptor": entry_inf.receptor_name,
            "connectionPoint": entry_inf.connection_pt.tolist()
        }

    def get_predictions_info(self, entry_idx: int):
        # Project averaged predictions into pca (or other) space.
        if self.averaged_predicted_fp_vis_rep_projected is None:
            self.averaged_predicted_fp_vis_rep_projected = self.vis_rep_space.project(self.predictions_ensembled)

        entry = {
            "averagedPrediction": {
                "vizRepProjection": self.averaged_predicted_fp_vis_rep_projected[entry_idx],
                "closestFromLabelSet": []
            },
            "predictionsPerRotation": [
                self.viz_reps_per_rotation[i][entry_idx].tolist()
                for i in range(self.num_rotations)
            ]
        }
        return entry

    @abstractmethod
    def _create_initial_prediction_tensor(self):
        pass

    @abstractmethod
    def _udpate_prediction_tensor(self, predicitons_to_add: torch.Tensor, idx: int):
        pass

    @abstractmethod
    def _finalize_prediction_tensor(self):
        # Should modify self.predictions_ensembled directly.
        pass
