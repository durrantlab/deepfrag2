from collagen.external.moad.moad import (
    MOAD_split,
    MOADInterface,
)
from torch.utils.data import Dataset
from typing import List, Dict, Union, Tuple, Set, Optional, Any, Callable
from ...core.mol import Mol
from pathlib import Path
import multiprocessing
from tqdm.auto import tqdm
import json
from dataclasses import dataclass


def build_frag_index_target(pdb_id):
    # Given a pdb_id, gets its ligands and maps those to the massess of all
    # associated fragments. Fragments are determined deterministically, so no
    # need to store fragment indexes.

    global MOAD_REF

    moad_entry_info = MOAD_REF[pdb_id]
    ligs_to_mass = []

    for i in range(len(moad_entry_info)):
        # lig_to_mass = {}
        lig_to_mass = []

        try:
            # Unpack info to get ligands
            _, ligands = moad_entry_info[i]

            for lig in ligands:
                # try:
                #     frag_masses = [int(x[1].mass) for x in lig.split_bonds()]
                # except:
                #     frag_masses = []
                # lig_to_mass[lig.meta["moad_ligand"].name] = int(lig.mass)
                lig_to_mass.append(lig.meta["moad_ligand"].name)
        except:
            pass

        ligs_to_mass.append(lig_to_mass)

    return (pdb_id, ligs_to_mass)


@dataclass
class MOADWholeLigDataset_entry(object):
    pdb_id: str
    lig_to_mass_chunk_idx: int
    ligand_id: str
    # frag_idx: int


class MOADWholeLigDataset(Dataset):
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

    # A cache-able index listing every fragment size for every ligand/target in the dataset.
    #
    # See MOADFragmentDataset._build_index for structure. This index only needs to be updated
    # for new structure files.
    _ligand_index_cached: Optional[dict]

    # The internal listing of every valid fragment example. This index is generated on each
    # run based on the runtime filters: (targets, smiles, fragment_size).
    _internal_index_valids_filtered: List[MOADWholeLigDataset_entry]

    def __init__(
        self,
        moad: MOADInterface,
        cache_file: Optional[Union[str, Path]] = None,
        cache_cores: int = 1,
        split: Optional[MOAD_split] = None,
        transform: Optional[Callable[[Mol, Mol, Mol], Any]] = None,
    ):
        self.moad = moad
        self.split = split if split is not None else moad.full_split()
        self.transform = transform
        self._index(cache_file, cache_cores)

    def _build_index(self, cores: int = 1) -> dict:
        """
        Cache format:

        {
            "pdb_id": [
                {
                    "lig_name": [fmass1, fmass2, ..., fmassN],
                    "lig_name2": [...]
                },
                ...
            ],
            ...
        }
        """
        global MOAD_REF
        MOAD_REF = self.moad

        index = {}
        pdb_ids_queue = self.moad.targets

        pbar = tqdm(total=len(pdb_ids_queue), desc="Building MOAD cache")
        with multiprocessing.Pool(cores) as p:
            for pdb_id, ligs_to_frag_masses in p.imap_unordered(
                build_frag_index_target, pdb_ids_queue
            ):
                index[pdb_id.lower()] = ligs_to_frag_masses
                pbar.update(1)

        pbar.close()
        return index

    def _index(self, cache_file: Optional[Union[str, Path]] = None, cores: int = 1):
        cache_file = Path(cache_file) if cache_file is not None else None

        if cache_file is not None and cache_file.exists():
            print("Loading MOAD fragments from cache...")
            with open(cache_file, "r") as f:
                index = json.load(f)
        else:
            index = self._build_index(cores)
            if cache_file is not None:
                with open(cache_file, "w") as f:
                    f.write(json.dumps(index))

        internal_index = []
        for pdb_id in tqdm(self.split.targets, desc="Runtime filters"):
            ligs_to_mass_chunks = index[pdb_id.lower()]
            for lig_to_mass_chunk_idx in range(len(ligs_to_mass_chunks)):
                lig_to_mass_chunk = ligs_to_mass_chunks[lig_to_mass_chunk_idx]
                # lig_to_frag_masses_chunk looks like:
                # {
                #   'ADN:A:901': [17, 31, 17, 17, 133, 16],
                #   'ADN:B:902': [17, 31, 17, 17, 133, 16],
                #   'ADN:C:903': [17, 31, 17, 17, 133, 16],
                #   'ADN:D:904': [17, 31, 17, 17, 133, 16],
                #   'ADN:E:905': [17, 31, 17, 17, 133, 16],
                #   'ADN:F:906': [17, 31, 17, 17, 133, 16]
                # }

                for ligand_id in lig_to_mass_chunk:
                    # Enforce SMILES filter.  TODO: Distance to receptor, number
                    # of heavy atoms, etc.?
                    skip = False
                    for lig in self.moad[pdb_id].ligands:
                        if (
                            lig.name == ligand_id
                            and lig.smiles not in self.split.smiles
                        ):
                            # You've found the ligand that is not in the split,
                            # so skip it.
                            skip = True
                            break

                    if skip:
                        continue

                    # lig_mass = lig_to_mass_chunk[ligand_id]
                    # if lig_mass != 0:
                    # A fragment with mass, so proceed.
                    internal_index.append(
                        MOADWholeLigDataset_entry(
                            pdb_id=pdb_id,
                            lig_to_mass_chunk_idx=lig_to_mass_chunk_idx,
                            ligand_id=ligand_id,
                            # frag_idx=frag_idx,
                        )
                    )
                    # print(lig_mass)
                    # for frag_idx in range(len(lig_mass)):
                    # if lig_mass[frag_idx] != 0:

        self._ligand_index_cached = index
        self._internal_index_valids_filtered = internal_index

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
