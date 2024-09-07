"""DeepFrag model that uses additional SDF data for training."""

from typing import List, Optional, TYPE_CHECKING
from apps.deepfrag.model import DeepFragModel
from collagen.external.common.types import StructureEntry
from collagen.external.paired_csv.interface import PairedCsvInterface
from collagen.metrics import cos_loss
from math import e

if TYPE_CHECKING:
    import torch  # type: ignore

class DeepFragModelPairedDataFinetune(DeepFragModel):
    
    """DeepFrag model that uses additional paired data for finetuning."""

    def __init__(self, **kwargs):
        """Initialize the DeepFrag model.
        
        Args:
            **kwargs: The arguments.
        """
        super().__init__(**kwargs)

        self.is_cpu = kwargs["cpu"]
        self.fragment_representation = kwargs["fragment_representation"]
        self.database = None
        self.use_prevalence = kwargs["use_prevalence"]

    def set_database(self, database: PairedCsvInterface):
        """Specify the paired database.

        Args:
            database (PairedCsvInterface): The paired database.
        """
        self.database = database

    def loss(self, pred: "torch.Tensor", fps: "torch.Tensor", entry_infos: Optional[List[StructureEntry]], batch_size: Optional[int]):
        """Loss function.

        Args:
            pred: tensor with the fingerprint values obtained from voxels.
            fps: tensor with the fingerprint values obtained from a given fragment representation.
            entry_infos (List[StructureEntry]): list with each entry information.
            batch_size: size of the tensors and list aforementioned.

        Returns:
            float: loss value
        """

        assert (
            self.database is not None
        ), "Database must be set before calling loss function."

        # Closer to 1 means more dissimilar, closer to 0 means more similar.
        if self.is_continuous_fingerprint:
            return super().loss(pred, fps, entry_infos, batch_size)

        cos_loss_vector = cos_loss(pred, fps)
        assert entry_infos is not None, "Entry infos must be provided."
        for idx, entry_info in enumerate(entry_infos):
            entry_data = self.database.frag_and_act_x_parent_x_sdf_x_pdb[
                entry_info.ligand_id
            ][entry_info.fragment_idx]
            act_value = float(entry_data[1])
            prv_value = (
                float(entry_data[3]) * -1 if self.use_prevalence else 0
            )  # considering neg prevalence

            # the lower the prevalence, the lower the result to raise euler to the prevalence.
            exp_value = e ** prv_value

            # the activity with the receptor is penalized
            # this increase makes its tendency to 0 more difficult when multiplying by the probability obtained from the cosine similarity function
            act_euler = act_value * exp_value
            cos_loss_vector[idx] = cos_loss_vector[idx] * act_euler

        return self.aggregation.aggregate_on_pytorch_tensor(cos_loss_vector)
