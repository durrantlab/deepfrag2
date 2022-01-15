from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Set, Optional, Any, Callable
from pathlib import Path

from torch.utils.data import Dataset

from .cache import CacheItemsToUpdate, build_index_and_filter
from ... import Mol


@dataclass
class MOADFragmentDataset_entry(object):
    pdb_id: str
    lig_to_frag_masses_chunk_idx: int
    ligand_id: str
    frag_idx: int


class MOADFragmentDataset(Dataset):
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

    moad: "MOADInterface"
    split: "MOAD_split"
    transform: Optional[Callable[[Mol, Mol, Mol], Any]]

    # A cache-able index listing every fragment size for every ligand/target in the dataset.
    #
    # See MOADFragmentDataset._build_index for structure. This index only needs to be updated
    # for new structure files.
    _fragment_index_cached: Optional[dict]

    # The internal listing of every valid fragment example. This index is generated on each
    # run based on the runtime filters: (targets, smiles, fragment_size).
    _internal_index_valids_filtered: List[MOADFragmentDataset_entry]

    def __init__(
        self,
        moad: "MOADInterface",
        cache_file: Optional[Union[str, Path]] = None,
        cache_cores: int = 1,
        split: Optional["MOAD_split"] = None,
        transform: Optional[Callable[[Mol, Mol, Mol], Any]] = None,
    ):
        self.moad = moad
        self.split = split if split is not None else moad.full_split()
        self.transform = transform
        self._index(cache_file, cache_cores)

    def _index(self, cache_file: Optional[Union[str, Path]] = None, cores: int = 1):
        def make_dataset_entries_func(
            pdb_id: str, lig_name: str, lig_inf: Dict
        ) -> List[MOADFragmentDataset_entry]:
            # Here also doing some filtering of the fragments.
            frag_masses = lig_inf["frag_masses"]
            entries_to_return = []
            for frag_idx in range(len(frag_masses)):
                if frag_masses[frag_idx] != 0:
                    # A fragment with mass, so proceed. TODO: JDD: More filters
                    # here.
                    entries_to_return.append(
                        MOADFragmentDataset_entry(
                            pdb_id=pdb_id,
                            lig_to_frag_masses_chunk_idx=lig_inf["lig_chunk_idx"],
                            ligand_id=lig_name,
                            frag_idx=frag_idx,
                        )
                    )
            return entries_to_return

        def lig_filter(lig: "MOAD_ligand", lig_inf: Dict) -> bool:
            # Filter applied to fragment, but not whole ligand.
            return True

        index, internal_index = build_index_and_filter(
            lig_filter,
            self.moad,
            self.split,
            make_dataset_entries_func,
            CacheItemsToUpdate(
                num_heavy_atoms=True,
                frag_masses=True,
                frag_dists_to_recep=True,
                frag_num_heavy_atoms=True,
                lig_mass=True,
                frag_smiles=True,  # Good for debugging.
                murcko_scaffold=True,
                
            ),
            cache_file,
            cores,
        )

        self._ligand_index_cached = index
        self._internal_index_valids_filtered = internal_index

    def __len__(self) -> int:
        return len(self._internal_index_valids_filtered)

    def __getitem__(self, idx: int) -> Tuple[Mol, Mol, Mol]:
        """Returns (receptor, parent, fragment)"""
        assert idx >= 0 and idx <= len(self), "Index out of bounds"

        entry = self._internal_index_valids_filtered[idx]

        receptor, ligands = self.moad[entry.pdb_id][entry.lig_to_frag_masses_chunk_idx]

        # This chunk has many ligands. You need to look up the one that matches
        # entry.ligand_id (the one that actually corresponds to this entry).
        # Once you find it, actually do the fragmenting.
        for ligand in ligands:
            if ligand.meta["moad_ligand"].name == entry.ligand_id:
                pairs = ligand.split_bonds()
                parent, fragment = pairs[entry.frag_idx]
                break
        else:
            raise Exception("Ligand not found")

        sample = (receptor, parent, fragment)

        if self.transform:
            # Actually performs voxelization and fingerprinting.
            return self.transform(sample)
        else:
            return sample
