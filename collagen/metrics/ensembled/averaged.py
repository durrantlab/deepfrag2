# ensemble averaged

import torch
from collagen.metrics.ensembled.parent import ParentEnsembled
import numpy as np
from apps.deepfrag.AggregationOperators import Operator
from apps.deepfrag.AggregationOperators import Aggregate1DTensor
import multiprocessing
# Given multiple predictions, this class can be used to average them. Much of
# the "meat" is in ParentEnsembled.


def _finalize_aggregation(dict_predictions_ensembled, aggregation, device, idx):
    nested_list = dict_predictions_ensembled[idx.__str__()]
    tensor_resp = np.zeros(len(nested_list[0]), dtype=float)
    matrix = np.matrix(nested_list)
    print("starting _finalize_aggregation " + str(idx))
    for j in range(0, len(tensor_resp)):
        tensor_resp[j] = aggregation.aggregate_on_numpy_array((np.asarray(matrix[:, j])).flatten())
    tensor_resp = torch.tensor(tensor_resp, dtype=torch.float32, device=device, requires_grad=False)
    print("finishing _finalize_aggregation " + str(idx))
    return idx, tensor_resp


class AveragedEnsembled(ParentEnsembled):

    dict_predictions_ensembled = {}
    aggregation = None

    def __init__(self, trainer, model, test_data, num_rotations, device, ckpt_name, aggregation_function: Operator):
        if aggregation_function != Operator.MEAN.value and num_rotations == 1:
            raise Exception("Use more than one rotation to use an aggregation function other than Mean (average) function")

        if aggregation_function != Operator.MEAN.value:
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
        print("Starting _finalize_prediction_tensor")
        if self.num_rotations == 1 or self.aggregation is None:
            self.predictions_ensembled = torch.tensor(self.predictions_ensembled, dtype=torch.float32, device=self.device, requires_grad=False)
            torch.div(
                self.predictions_ensembled,
                torch.tensor(self.num_rotations, dtype=torch.float32, device=self.device, requires_grad=False),
                out=self.predictions_ensembled
            )
        else:
            print("Starting aggregation")
            parameters = [(self.dict_predictions_ensembled, self.aggregation, self.device, i) for i in range(0, len(self.dict_predictions_ensembled))]
            with multiprocessing.Pool() as p:
                for i, tensor_resp in p.starmap(_finalize_aggregation, iterable=parameters, chunksize=1):
                    print("starting assignment " + str(i))
                    self.predictions_ensembled[i] = tensor_resp
                    print("finishing assignment " + str(i))
                p.close()
            print("Finishing aggregation")
        print("Finishing _finalize_prediction_tensor")
