# ensemble averaged

from typing import Any
import torch
from collagen.metrics.ensembled.parent import ParentEnsembled

# Given multiple predictions, this class can be used to average them. Much of
# the "meat" is in ParentEnsembled.

class AveragedEnsembled(ParentEnsembled):
    def _create_initial_prediction_tensor(self) -> torch.Tensor:
        # At this point, model is after inference on the first rotation. So it
        # has a prediciton.
        return self.model.predictions.detach().clone()

    def _udpate_prediction_tensor(self, predicitons_to_add: torch.Tensor, idx: int):
        torch.add(self.predictions_ensembled, predicitons_to_add, out=self.predictions_ensembled)

    def _finalize_prediction_tensor(self):
        # Divide by number of rotations to get the final average predicitons.
        torch.div(
            self.predictions_ensembled, 
            torch.tensor(self.num_rotations, device=self.device),
            out=self.predictions_ensembled
        )
