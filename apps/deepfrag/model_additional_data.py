from apps.deepfrag.model import DeepFragModel
from collagen.external.moad.interface import SdfDirInterface
from collagen.external.moad.split import compute_dataset_split
from collagen.external.moad.cache_filter import CacheItemsToUpdate, get_info_given_pdb_id
from collagen.core.molecules.fingerprints import fingerprint_for
from rdkit import Chem
import numpy as np
import random
import torch


class DeepFragModelSDFData(DeepFragModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.is_cpu = kwargs["cpu"]
        self.fragment_representation = kwargs["fragment_representation"]
        self.fragments = DeepFragModelSDFData.__get_train_val_sdf_sets(**kwargs)

    def loss(self, pred, fps, entry_infos, batch_size):
        batch_size = fps.shape[0]
        selected_fragments = random.choices(self.fragments, k=batch_size)
        fps_bad = np.zeros(shape=[batch_size, self.fp_size])

        idx = 0
        for c_fragment in selected_fragments:
            fps_bad[idx] = fingerprint_for(Chem.MolFromSmiles(c_fragment), self.fragment_representation, self.fp_size, c_fragment)
            idx = idx + 1

        fps_bad = torch.tensor(fps_bad, dtype=torch.float32, device=torch.device("cpu") if self.is_cpu else torch.device("cuda"), requires_grad=False)
        loss_1 = super().loss(pred, fps, entry_infos, batch_size)
        loss_2 = super().loss(pred, fps_bad, None, None)
        loss = loss_1 + (1 - loss_2)
        return loss

    @staticmethod
    def __get_train_val_sdf_sets(**kwargs):

        print("Building/updating additional SDF dataset to train DeepFrag")
        sdf_data = SdfDirInterface(
            structures=kwargs["additional_training_data_dir"],
            cache_pdbs_to_disk=None,
            grid_width=None,
            grid_resolution=None,
            noh=None,
            discard_distant_atoms=kwargs["discard_distant_atoms"]
        )

        train, _, _ = compute_dataset_split(
            sdf_data,
            seed=kwargs["split_seed"],
            fraction_train=1.0,
            fraction_val=0.0,
            save_splits=None,
            load_splits=None,
            max_pdbs_train=None,
            max_pdbs_val=None,
            max_pdbs_test=None,
            butina_cluster_cutoff=None,
        )

        _, lig_infs = get_info_given_pdb_id(["non", sdf_data["non"], CacheItemsToUpdate(
            lig_mass=True,
            murcko_scaffold=True,
            num_heavy_atoms=True,
            frag_masses=True,
            frag_num_heavy_atoms=True,
            frag_dists_to_recep=False,
            frag_smiles=True,  # Good for debugging.
            frag_aromatic=True,
            frag_charged=True,
        ), train.smiles])

        fragments = set()
        for ligand_id in lig_infs:
            fragments_lig = lig_infs.get(ligand_id).get("frag_smiles")
            fragments.update(fragments_lig)
        return list(fragments)
