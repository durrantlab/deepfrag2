"""Given multiple predictions, this class can be used to average them. Much of
the "meat" is in ParentEnsembled.
"""

from collagen.core.loader import DataLambda
import torch  # type: ignore
from collagen.metrics.ensembled.parent import ParentEnsembled
import numpy as np  # type: ignore
from collagen.apps.deepfrag.AggregationOperators import Operator
from collagen.apps.deepfrag.AggregationOperators import Aggregate1DTensor
import os.path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytorch_lightning as pl  # type: ignore


class AveragedEnsembled(ParentEnsembled):

    """AveragedEnsembled is a class that can be used to average multiple
    predictions. It is a child class of ParentEnsembled.
    """

    dict_predictions_ensembled = {}
    aggregation = None

    def __init__(
        self,
        trainer: "pl.Trainer",
        model: "pl.LightningModule",
        test_data: DataLambda,
        num_rotations: int,
        device: torch.device,
        ckpt_name: str,
        aggregation_function: Operator,
        frag_representation: str,
        save_fps=False,
    ):
        """Initialize the class.

        Args:
            trainer (pl.Trainer): The trainer object.
            model (pl.LightningModule): The model object.
            test_data (DataLambda): The test data.
            num_rotations (int): The number of rotations to perform.
            device (torch.device): The device to use.
            ckpt_name (str): The name of the checkpoint.
            aggregation_function (Operator): The aggregation function to use.
            frag_representation (str): The name of the fragment representation.
            save_fps (bool): Whether to save the fingerprints.
        """
        self.frag_representation = frag_representation
        self.save_fps = save_fps

        if aggregation_function != Operator.MEAN.value and num_rotations == 1:
            raise Exception(
                "Use more than one rotation to use an aggregation function other than Mean (average) function"
            )

        if aggregation_function != Operator.MEAN.value:
            self.aggregation = Aggregate1DTensor(operator=aggregation_function)

        ParentEnsembled.__init__(
            self, trainer, model, test_data, num_rotations, device, ckpt_name
        )

    def _create_initial_prediction_tensor(self) -> torch.Tensor:
        """Create the initial prediction tensor. This is the tensor that will
        hold the predictions from all the rotations. At this point, model is
        after inference on the first rotation. So it has a prediciton.

        Returns:
            torch.Tensor: The initial prediction tensor.
        """
        predictions_to_return = self.model.predictions.detach().clone()

        if self.num_rotations > 1 and self.aggregation is not None:
            predictions_ = predictions_to_return.cpu().detach().clone()
            self.dict_predictions_ensembled = {
                i.__str__(): [predictions_[i].numpy()] for i in range(len(predictions_))
            }

        return predictions_to_return

    def _udpate_prediction_tensor(self, predicitons_to_add: torch.Tensor, idx: int):
        """Add the predictions from the current rotation to the predictions
        tensor (in place).

        Args:
            predicitons_to_add (torch.Tensor): The predictions from the current
                rotation.
            idx (int): The index of the current rotation.
        """
        if self.num_rotations > 1 and self.aggregation is not None:
            predictions_ = predicitons_to_add.cpu().detach().clone()
            for i in range(len(predictions_)):
                self.dict_predictions_ensembled[i.__str__()].append(
                    predictions_[i].numpy()
                )

        torch.add(
            self.predictions_ensembled,
            predicitons_to_add,
            out=self.predictions_ensembled,
        )

    def _finalize_prediction_tensor(self):
        """Divide the predictions tensor by the number of rotations to get the
        average.
        """
        # TODO: Are we using the alternate aggregation schemes?

        if self.num_rotations == 1 or self.aggregation is None:
            # If there is only one rotation, or if we are using the mean
            # function, then we can just divide by the number of rotations.

            self.predictions_ensembled = torch.tensor(
                self.predictions_ensembled,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            torch.div(
                self.predictions_ensembled,
                torch.tensor(
                    self.num_rotations,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False,
                ),
                out=self.predictions_ensembled,
            )
        else:
            # If we are using an aggregation function other than mean, then we
            # need to apply the aggregation function to each column of the
            # tensor.

            for i in range(len(self.dict_predictions_ensembled)):
                nested_list = self.dict_predictions_ensembled[i.__str__()]
                tensor_resp = np.zeros(len(nested_list[0]), dtype=float)
                matrix = np.matrix(nested_list)
                for j in range(len(tensor_resp)):
                    array_col = np.asarray(matrix[:, j])
                    array_col = array_col.flatten()
                    tensor_resp[j] = self.aggregation.aggregate_on_numpy_array(
                        array_col
                    )
                tensor_resp = torch.tensor(
                    tensor_resp,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False,
                )
                self.predictions_ensembled[i] = tensor_resp

            self.predictions_ensembled = torch.tensor(
                self.predictions_ensembled,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )

        if self.save_fps:
            frag_fps = {}
            recep_parent_fps = {}

            for idx, entry in enumerate(self.model.prediction_targets_entry_infos):

                if entry.ligand_id not in recep_parent_fps.keys():
                    recep_parent_fps[entry.ligand_id] = self.predictions_ensembled[idx]

                if entry.fragment_smiles not in frag_fps.keys():
                    frag_fps[entry.fragment_smiles] = self.model.prediction_targets[idx]

            torch.save(
                recep_parent_fps,
                os.path.realpath(os.getcwd())
                + os.sep
                + self.frag_representation
                + "_pred_fingerprints.pt",
            )
            torch.save(
                frag_fps,
                os.path.realpath(os.getcwd())
                + os.sep
                + self.frag_representation
                + "_calc_fingerprints.pt",
            )
