"""Run DeepFrag."""

from apps.deepfrag.run import DeepFrag
from collagen.external.common.datasets.fragment_dataset import FragmentDataset
from collagen.external.common.types import StructureEntryForMultimodal
from apps.deepfrag.model_fusing_modalities import DeepFragModelESM2


class MultimodalDeepFrag(DeepFrag):

    """DeepFrag model."""

    def __init__(self):
        """Initialize the DeepFrag model parent."""

        super().__init__(model_cls=DeepFragModelESM2, dataset_cls=FragmentDataset)

    def _get_payload(self, rec, parent, frag, ligand_id, fragment_idx, center):
        frag_smiles = frag.smiles(True)
        parent_smiles = parent.smiles(True)
        receptor_name = rec.meta['name']
        receptor_sequence = rec.aminoacid_sequence().upper()

        assert (frag_smiles is not None and parent_smiles is not None and receptor_sequence is not None), \
            f"Fragment ({frag_smiles}) or parent ({parent_smiles}) SMILES, or Receptor sequence ({receptor_name}) " \
            f"is None"

        return StructureEntryForMultimodal(
            fragment_smiles=frag_smiles,
            parent_smiles=parent_smiles,
            receptor_name=receptor_name,
            receptor_sequence=receptor_sequence,
            connection_pt=center,
            ligand_id=ligand_id,
            fragment_idx=fragment_idx,
        )
