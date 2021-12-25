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


# def build_whole_lig_index_target(packed):
#     global MOAD_REF

#     pdb = packed
#     target = MOAD_REF[pdb]
#     samples = []

#     for i in range(len(target)):
#         sample = {}

#         # try:
#         _, ligands = target[i]

#         for lig in ligands:
#             # try:
#             #     frags = [int(x[1].mass) for x in lig.split_bonds()]
#             # except:
#             #     frags = []
#             sample[lig.meta["moad_ligand"].name] = Mol.from_rdkit(lig.rdmol)

#         # except:
#             # pass

#         samples.append(sample)

#     return (pdb, samples)


@dataclass
class MOADWholeLigDataset_entry(object):
    pdb_id: str
    # sample: int
    ligand: str
    # frag_idx: int


class MOADWholeLigDataset(Dataset):
    """
    A Dataset that provides (receptor, ligand) tuples.

    Args:
        moad (MOADInterface): An initialized MOADInterface object.
        cache_file (str, optional): Path to a cache file to store or load fragment metadata.
        cache_cores (int, optional): If a cache file is not found, use this many cores to compute a new cache.
        split (MOAD_split, optional): An optional split to constrain the space of examples.
        transform (Callable[[Mol, Mol], Any], optional): An optional transformation function to invoke before returning samples.
            Takes the arguments (receptor, ligand) as Mol objects.
    """

    moad: MOADInterface
    split: MOAD_split
    transform: Optional[Callable[[Mol, Mol], Any]]

    # A cache-able index listing every fragment size for every ligand/target in the dataset.
    #
    # See MOADFragmentDataset._build_index for structure. This index only needs to be updated
    # for new structure files.
    _fragment_index: Optional[dict]

    # The internal listing of every valid fragment example. This index is generated on each
    # run based on the runtime filters: (targets, smiles, fragment_size).
    _internal_index: List[MOADWholeLigDataset_entry]

    def __init__(
        self,
        moad: MOADInterface,
        cache_file: Optional[Union[str, Path]] = None,
        cache_cores: int = 1,
        split: Optional[MOAD_split] = None,
        transform: Optional[Callable[[Mol, Mol], Any]] = None,
    ):
        self.moad = moad
        self.split = split if split is not None else moad.full_split()
        self.transform = transform
        self._index(cache_file, cache_cores)

    # def _build_index(self, cores: int = 1) -> dict:
    #     """
    #     Cache format:

    #     {
    #         "pdb_id": [
    #             {
    #                 "lig_name": [fmass1, fmass2, ..., fmassN],
    #                 "lig_name2": [...]
    #             },
    #             ...
    #         ],
    #         ...
    #     }
    #     """
    #     global MOAD_REF
    #     MOAD_REF = self.moad

    #     index = {}
    #     queue = self.moad.targets

    #     pbar = tqdm(total=len(queue), desc="Building MOAD cache")
    #     with multiprocessing.Pool(cores) as p:
    #         for pdb, item in p.imap_unordered(build_whole_lig_index_target, queue):
    #             index[pdb.lower()] = item
    #             pbar.update(1)

    #     pbar.close()
    #     return index

    def _index(self, cache_file: Optional[Union[str, Path]] = None, cores: int = 1):
        cache_file = Path(cache_file) if cache_file is not None else None

        # if cache_file is not None and cache_file.exists():
        #     print("Loading MOAD fragments from cache...")
        #     with open(cache_file, "r") as f:
        #         index = json.load(f)
        # else:
        #     index = self._build_index(cores)
        #     import pdb; pdb.set_trace()
        #     if cache_file is not None:
        #         with open(cache_file, "w") as f:
        #             f.write(json.dumps(index))

        internal_index = []
        for pdb_id in tqdm(self.split.targets, desc="Runtime filters"):
            for ligand in self.moad[pdb_id].ligands:
                # Enforce SMILES filter.
                if ligand.smiles not in self.split.smiles:
                    continue
                
                internal_index.append(
                    MOADWholeLigDataset_entry(
                        pdb_id=pdb_id,
                        # sample=sample,
                        ligand=ligand,
                        # frag_idx=frag_idx,
                    )
                )
                # import pdb; pdb.set_trace()

        #     samples = index[pdb_id.lower()]
        #     for sample in range(len(samples)):
        #         s = samples[sample]
        #         for ligand in s:
        #             # Enforce SMILES filter.
        #             skip = False
        #             for lig in self.moad[pdb_id].ligands:
        #                 if lig.name == ligand and lig.smiles not in self.split.smiles:
        #                     skip = True
        #                     break

        #             if skip:
        #                 continue

        #             frags = s[ligand]
        #             for frag_idx in range(len(frags)):
        #                 if frags[frag_idx] != 0:
        #                     internal_index.append(
        #                         MOADWholeLigDataset_entry(
        #                             pdb_id=pdb_id,
        #                             sample=sample,
        #                             ligand=ligand,
        #                             frag_idx=frag_idx,
        #                         )
        #                     )

        # self._fragment_index = index
        self._internal_index = internal_index

    def __len__(self) -> int:
        return len(self._internal_index)

    def __getitem__(self, idx: int) -> Tuple[Mol, Mol]:
        """Returns (receptor, ligand)"""
        assert idx >= 0 and idx <= len(self), "Index out of bounds"

        entry = self._internal_index[idx]
        receptor, ligands = self.moad[entry.pdb_id][0]
        # receptor, ligand = self.moad[entry.pdb_id][entry.sample]

        for ligand in ligands:
            if ligand.meta["moad_ligand"].name == entry.ligand.name:
                # ligand found!
                print("found")
                break
        else:
            print("not found", entry.ligand.name, "in", [ligand.meta["moad_ligand"].name for ligand in ligands])
            # print("***", self.moad[entry.pdb_id][1])  # Never exists, I think.
            raise Exception("Ligand not found")

        sample = (receptor, ligand)

        # import pdb; pdb.set_trace()

        if self.transform:
            return self.transform(*sample)
        else:
            return sample
