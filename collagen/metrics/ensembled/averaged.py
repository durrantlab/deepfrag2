# ensemble averaged

import torch
from collagen.metrics.ensembled.parent import ParentEnsembled
import numpy as np
from apps.deepfrag.AggregationOperators import Operator
from apps.deepfrag.AggregationOperators import Aggregate1DTensor
# Given multiple predictions, this class can be used to average them. Much of
# the "meat" is in ParentEnsembled.


class AveragedEnsembled(ParentEnsembled):

    dict_predictions_ensembled = {}
    aggregation = None

    def __init__(self, trainer, model, test_data, num_rotations, device, ckpt_name, aggregation_function: Operator):
        if aggregation_function.__str__() != Operator.MEAN.__str__() and num_rotations == 1:
            raise "Use more than one rotation to use an aggregation function other than Mean (average) function"

        if aggregation_function.value != Operator.MEAN.value:
            self.aggregation = Aggregate1DTensor(operator=aggregation_function)

        ParentEnsembled.__init__(self, trainer, model, test_data, num_rotations, device, ckpt_name)

    def _create_initial_prediction_tensor(self) -> torch.Tensor:
        # At this point, model is after inference on the first rotation. So it
        # has a prediciton.
        predictions_ = self.model.predictions.detach().clone()
        if self.num_rotations > 1 and self.aggregation is not None:
            self.dict_predictions_ensembled = {i.__str__(): [predictions_[i].numpy()] for i in range(0, len(predictions_))}
        return predictions_

    def _udpate_prediction_tensor(self, predicitons_to_add: torch.Tensor, idx: int):
        if self.num_rotations > 1 and self.aggregation is not None:
            predictions_ = predicitons_to_add.detach().clone()
            for i in range(0, len(predictions_)):
                self.dict_predictions_ensembled[i.__str__()].append(predictions_[i].numpy())

        torch.add(self.predictions_ensembled, predicitons_to_add, out=self.predictions_ensembled)

    def _finalize_prediction_tensor(self):
        # Divide by number of rotations to get the final average predictions.
        if self.num_rotations == 1 or self.aggregation is None:
            torch.div(
                self.predictions_ensembled,
                torch.tensor(self.num_rotations, device=self.device),
                out=self.predictions_ensembled
            )
        else:
            for i in range(0, len(self.dict_predictions_ensembled)):
                nested_list = self.dict_predictions_ensembled[i.__str__()]
                tensor_resp = np.zeros(len(nested_list[0]), dtype=float)
                matrix = np.matrix(nested_list)
                for j in range(0, len(tensor_resp)):
                    tensor_resp[j] = self.aggregation.aggregate_on_numpy_array((np.asarray(matrix[:, j])).flatten())
                tensor_resp = torch.tensor(tensor_resp, dtype=torch.float32, requires_grad=False)
                self.predictions_ensembled[i] = tensor_resp
