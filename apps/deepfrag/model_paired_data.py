"""DeepFrag model that uses additional SDF data for training."""

from apps.deepfrag.model import DeepFragModel
from collagen.metrics import cos_loss
from math import e


class DeepFragModelPairedDataFinetune(DeepFragModel):

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

    def set_database(self, database):
        """Method to specify the paired database.

        Args:
            database: The paired database.
        """
        self.database = database

    def loss(self, pred, fps, entry_infos, batch_size):
        """Loss function.

        Args:
            pred: tensor with the fingerprint values obtained from voxels.
            fps: tensor with the fingerprint values obtained from a given fragment representation.
            entry_infos: list with each entry information.
            batch_size: size of the tensors and list aforementioned.

        Returns:
            float: loss value
        """
        # Closer to 1 means more dissimilar, closer to 0 means more similar.
        if self.is_regression_mode:
            return super().loss(pred, fps, entry_infos, batch_size)

        cos_loss_vector = cos_loss(pred, fps)
        for idx, entry in enumerate(entry_infos):
            entry_data = self.database.frag_and_act_x_parent_x_sdf_x_pdb[entry.ligand_id][entry.fragment_idx]
            act_value = float(entry_data[1])
            prv_value = float(entry_data[3]) * -1 if self.use_prevalence else 0  # considering neg prevalence

            # the lower the prevalence, the lower the result to raise euler to the prevalence.
            exp_value = e ** prv_value

            # the activity with the receptor is penalized
            # this increase makes its tendency to 0 more difficult when multiplying by the probability obtained from the cosine similarity function
            act_euler = act_value * exp_value
            cos_loss_vector[idx] = cos_loss_vector[idx] * act_euler

        return self.aggregation.aggregate_on_pytorch_tensor(cos_loss_vector)
