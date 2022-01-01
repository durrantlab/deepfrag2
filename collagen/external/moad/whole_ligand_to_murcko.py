from collagen.external.moad.moad import (
    MOAD_ligand,
    MOAD_split,
    build_index_and_filter,
)
from collagen.external.moad.moad_interface import MOADInterface
from torch.utils.data import Dataset
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
from ...core.mol import Mol
from pathlib import Path
from dataclasses import dataclass


@dataclass
class MOADMurckoDataset_entry(object):
    pdb_id: str
    lig_to_mass_chunk_idx: int
    ligand_id: str
    # frag_idx: int


class MOADMurckoLigDataset(Dataset):
    """
    A Dataset that provides (receptor, parent, fragment) tuples by splitting ligands on single bonds.

    Args:
        moad (MOADInterface): An initialized MOADInterface object.
        cache_file (str, optional): Path to a cache file to store or load fragment metadata.
        cache_cores (int, optional): If a cache file is not found, use this many cores to compute a new cache.
        split (MOAD_split, optional): An optional split to constrain the space of examples.
        transform (Callable[[Mol, Mol, Mol], Any], optional): An optional transformation function to invoke before returning samples.
            Takes the arguments (receptor, parent, fragment) as Mol objects.
    """

    moad: MOADInterface
    split: MOAD_split
    transform: Optional[Callable[[Mol, Mol, Mol], Any]]

    # A cache-able index listing every fragment size for every ligand/target in
    # the dataset.
    #
    # See MOADFragmentDataset._build_index for structure. This index only needs
    # to be updated for new structure files.
    _ligand_index_cached: Optional[dict]

    # The internal listing of every valid fragment example. This index is
    # generated on each run based on the runtime filters: (targets, smiles,
    # fragment_size).
    _internal_index_valids_filtered: List[MOADMurckoDataset_entry]

    def __init__(
        self,
        moad: MOADInterface,
        cache_file: Optional[Union[str, Path]] = None,
        cache_cores: int = None,
        split: Optional[MOAD_split] = None,
        transform: Optional[Callable[[Mol, Mol, Mol], Any]] = None,
    ):
        self.moad = moad
        self.split = split if split is not None else moad.full_split()
        self.transform = transform
        self._index(cache_file, cache_cores)

    def _index(self, cache_file: Optional[Union[str, Path]] = None, cores: int = 1):
        def make_dataset_entry_func(pdb_id: str, lig_name: str, lig_inf: Dict):
            MOADMurckoDataset_entry(
                pdb_id=pdb_id,
                lig_to_mass_chunk_idx=lig_inf["lig_chunk_idx"],
                ligand_id=lig_name,
                # frag_idx=frag_idx,
            )

        def filter(lig: MOAD_ligand, lig_inf: Dict) -> bool:
            if lig_inf["lig_mass"] > 500:
                # Lignad is too big.
                return False

            # smi_fixed = fix_moad_smiles(lig.smiles)
            # # if lig.smiles == "":
            # # import pdb; pdb.set_trace()
            # print(lig.name)
            # print("*" + smi_fixed)
            # print(MurckoScaffoldSmilesFromSmiles(smiles=smi_fixed, includeChirality=True))
            # print("===")
            # print(MurckoScaffoldSmilesFromSmiles(lig.smiles, includeChirality=True))
            # if MurckoScaffoldSmilesFromSmiles(lig.smiles, includeChirality=True) == "":
            #     # It has no mercko scaffold, so skip.
            #     skip = True
            #     break

            return True

        index, internal_index = build_index_and_filter(
            filter, self.moad, self.split, make_dataset_entry_func, cache_file, cores
        )
        self._ligand_index_cached = index
        self._internal_index_valids_filtered = internal_index

        # cache_file = Path(cache_file) if cache_file is not None else None

        # index = build_moad_cache(cache_file, self.moad, lig_mass=True, cores=cores)

        # internal_index = []
        # for pdb_id in tqdm(self.split.targets, desc="Runtime filters"):
        #     pdb_id = pdb_id.lower()
        #     receptor_inf = index[pdb_id]
        #     for lig_name in receptor_inf.keys():
        #         lig_inf = receptor_inf[lig_name]

        #         # Enforce SMILES filter. TODO: Distance to receptor, number of
        #         # heavy atoms, etc.?
        #         skip = False
        #         for lig in self.moad[pdb_id].ligands:
        #             if lig.name == lig_name:
        #                 # You've found the ligand.
        #                 if lig.smiles not in self.split.smiles:
        #                     # It is not in the split, so skip it.
        #                     skip = True
        #                     break

        #                 if lig_inf["lig_mass"] > 500:
        #                     # Lignad is too big.
        #                     skip = True
        #                     break

        #                 # smi_fixed = fix_moad_smiles(lig.smiles)
        #                 # # if lig.smiles == "":
        #                 # # import pdb; pdb.set_trace()
        #                 # print(lig.name)
        #                 # print("*" + smi_fixed)
        #                 # print(MurckoScaffoldSmilesFromSmiles(smiles=smi_fixed, includeChirality=True))
        #                 # print("===")
        #                 # print(MurckoScaffoldSmilesFromSmiles(lig.smiles, includeChirality=True))
        #                 # if MurckoScaffoldSmilesFromSmiles(lig.smiles, includeChirality=True) == "":
        #                 #     # It has no mercko scaffold, so skip.
        #                 #     skip = True
        #                 #     break

        #         if skip:
        #             continue

        #         # lig_mass = lig_to_mass_chunk[ligand_id]
        #         # if lig_mass != 0:
        #         # A fragment with mass, so proceed.
        #         internal_index.append(
        #             MOADMurckoDataset_entry(
        #                 pdb_id=pdb_id,
        #                 lig_to_mass_chunk_idx=lig_inf["lig_chunk_idx"],
        #                 ligand_id=lig_name,
        #                 # frag_idx=frag_idx,
        #             )
        #         )

        # self._ligand_index_cached = index
        # self._internal_index_valids_filtered = internal_index

    def __len__(self) -> int:
        return len(self._internal_index_valids_filtered)

    def __getitem__(self, idx: int) -> Tuple[Mol, Mol, Mol]:
        """Returns (receptor, parent, fragment)"""
        assert idx >= 0 and idx <= len(self), "Index out of bounds"

        entry = self._internal_index_valids_filtered[idx]

        receptor, ligands = self.moad[entry.pdb_id][entry.lig_to_mass_chunk_idx]

        # This chunk has many ligands. You need to look up the one that matches
        # entry.ligand_id (the one that actually corresponds to this entry).
        # Once you find it, actually do the fragmenting.
        for ligand in ligands:
            if ligand.meta["moad_ligand"].name == entry.ligand_id:
                # pairs = ligand.split_bonds()
                # parent, fragment = pairs[entry.frag_idx]
                sample = (receptor, ligand)
                break
        else:
            raise Exception("Ligand not found")

        # print(receptor,  parent.pdb(), fragment.pdb())

        if self.transform:
            # Actually performs voxelization and fingerprinting.
            return self.transform(*sample)
        else:
            return sample
