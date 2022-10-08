import argparse
from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Set, Optional, Any, Callable
from pathlib import Path
from torch.utils.data import Dataset
from collagen.core import args as user_args
from collagen.external.moad.split import full_moad_split
from ..cache_filter import CacheItemsToUpdate, load_cache_and_filter
from .... import Mol
import sys


@dataclass
class MOADFragmentDataset_entry(object):
    pdb_id: str
    lig_to_frag_masses_chunk_idx: int
    ligand_id: str
    frag_idx: int


class MOADFragmentDataset(Dataset):
    """
    A Dataset that provides (receptor, parent, fragment) tuples by splitting
    ligands on single bonds. Used in DeepFrag, for example.

    Args:
        moad (MOADInterface): An initialized MOADInterface object.
        cache_file (str, optional): Path to a cache file to store or load
            fragment metadata.
        cache_cores (int, optional): If a cache file is not found, use this
            many cores to compute a new cache.
        split (MOAD_split, optional): An optional split to constrain the space
            of examples.
        transform (Callable[[Mol, Mol, Mol], Any], optional): An optional
            transformation function to invoke before returning samples. Takes
            the arguments (receptor, parent, fragment) as Mol objects.
    """

    moad: "MOADInterface"
    split: "MOAD_split"

    # function that performs voxelization and fingerprinting
    transform: Optional[Callable[[Mol, Mol, Mol], Any]]

    # A cache-able index listing every fragment size for every ligand/target in
    # the dataset.
    #
    # See MOADFragmentDataset._build_index for structure. This index only needs
    # to be updated for new structure files. TODO: Is _fragment_index_cached
    # even used?
    # _fragment_index_cached: Optional[dict]

    # The internal listing of every valid fragment example. This index is
    # generated on each run based on the runtime filters: (targets, smiles,
    # fragment_size).
    _internal_index_valids_filtered: List[MOADFragmentDataset_entry]

    def __init__(
        self,
        moad: "MOADInterface",
        cache_file: Optional[Union[str, Path]] = None,
        cache_cores: int = 1,
        split: Optional["MOAD_split"] = None,
        transform: Optional[Callable[[Mol, Mol, Mol], Any]] = None,
        args: argparse.Namespace = None,
    ):
        self.moad = moad
        self.split = split if split is not None else full_moad_split(moad)
        self.transform = transform
        self.args = args
        self._index(cache_file, cache_cores)

    @staticmethod
    def add_fragment_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        # Adds user arguments so user can control how fragments are generated.

        parser = parent_parser.add_argument_group("Fragment Dataset")

        parser.add_argument(
            "--min_frag_mass",
            required=False,
            type=float,
            default=0,
            help="Consider only fragments with at least this molecular mass. Default is 0 Da.",
        )
        parser.add_argument(
            "--max_frag_mass",
            required=False,
            type=float,
            default=150,
            help="Consider only fragments with at most this molecular mass. Default is 150 Da.",
        )
        parser.add_argument(
            "--max_frag_dist_to_recep",
            required=False,
            type=float,
            default=4,
            help="Consider only fragments that have at least one atom that comes within this distance of any receptor atom. Default is 4 Å.",
        )
        parser.add_argument(
            "--min_frag_num_heavy_atoms",
            required=False,
            type=int,
            default=1,
            help="Consider only fragments that have at least this number of heavy atoms. Default is 1.",
        )

        return parent_parser

    def _lig_filter(
        self, args: argparse.Namespace, lig: "MOAD_ligand", lig_inf: Dict
    ) -> bool:
        # In the case of the fragment dataset, there is a filter applied to
        # fragments, but not whole ligands. So everything passes. This is what
        # is passed to cache_filter.load_cache_and_filter as the lig_filter_func
        # parameter.
        return True

    def _frag_filter(
        self,
        args: argparse.Namespace,
        mass: float,
        frag_dist_to_recep: float,
        frag_num_heavy_atom: int,
    ) -> bool:
        # This filter is passed to cache_filter.load_cache_and_filter via the
        # make_dataset_entries_func parameter.

        if mass < args.min_frag_mass:
            # A fragment with no mass, so skip.
            if user_args.verbose:
                print(f"Fragment rejected; mass too small: {mass}")
            return False
        if mass > args.max_frag_mass:
            if user_args.verbose:
                print(f"Fragment rejected; mass too large: {mass}")
            return False
        if frag_dist_to_recep > args.max_frag_dist_to_recep:
            if user_args.verbose:
                print(
                    f"Fragment rejected; distance from receptor too large: {frag_dist_to_recep}"
                )

            return False
        if frag_num_heavy_atom < args.min_frag_num_heavy_atoms:
            if user_args.verbose:
                print(
                    f"Fragment rejected; has too few heavy atoms: {frag_num_heavy_atom}"
                )
            return False
        return True

    def _make_dataset_entries_func(
        self, args: argparse.Namespace, pdb_id: str, lig_name: str, lig_inf: Dict
    ) -> List[MOADFragmentDataset_entry]:
        # This filter is passed to cache_filter.load_cache_and_filter as the
        # make_dataset_entries_func parameter.

        # Note that lig_inf contains all the data from the cache.

        # Here also doing some filtering of the fragments.
        frag_masses = lig_inf["frag_masses"]
        frag_dists_to_recep = lig_inf["frag_dists_to_recep"]
        frag_num_heavy_atoms = lig_inf["frag_num_heavy_atoms"]

        entries_to_return = []
        for frag_idx in range(len(frag_masses)):
            mass = frag_masses[frag_idx]
            dist_to_recep = frag_dists_to_recep[frag_idx]
            num_heavy_atom = frag_num_heavy_atoms[frag_idx]
            if self._frag_filter(args, mass, dist_to_recep, num_heavy_atom):
                entries_to_return.append(
                    MOADFragmentDataset_entry(
                        pdb_id=pdb_id,
                        lig_to_frag_masses_chunk_idx=lig_inf["lig_chunk_idx"],
                        ligand_id=lig_name,
                        frag_idx=frag_idx,
                    )
                )
        return entries_to_return

    def _index(self, cache_file: Optional[Union[str, Path]] = None, cores: int = 1):
        # Creates the cache and filtered cache, here referred to as an index.

        cache, filtered_cache = load_cache_and_filter(
            self._lig_filter,
            self.moad,
            self.split,
            self.args,
            self._make_dataset_entries_func,
            CacheItemsToUpdate(
                lig_mass=True,
                murcko_scaffold=True,
                num_heavy_atoms=True,
                frag_masses=True,
                frag_num_heavy_atoms=True,
                frag_dists_to_recep=True,
                frag_smiles=True,  # Good for debugging.
            ),
            cache_file,
            cores,
        )

        self._ligand_index_cached = cache
        self._internal_index_valids_filtered = filtered_cache

    def __len__(self) -> int:
        return len(self._internal_index_valids_filtered)

    def __getitem__(self, idx: int) -> Tuple[Mol, Mol, Mol]:
        """Returns (receptor, parent, fragment)"""
        assert 0 <= idx <= len(self), "Index out of bounds"
        entry = None
        try:
            entry = self._internal_index_valids_filtered[idx]

            receptor, ligands = self.moad[entry.pdb_id][entry.lig_to_frag_masses_chunk_idx]

            # with open("/mnt/extra/fragz2.txt", "a") as f:
            #     f.write(receptor.meta["name"] + "\t" + str(ligands) + "\n")

            # This chunk has many ligands. You need to look up the one that matches
            # entry.ligand_id (the one that actually corresponds to this entry).
            # Once you find it, actually do the fragmenting.
            for ligand in ligands:
                if ligand.meta["moad_ligand"].name == entry.ligand_id:
                    pairs = ligand.split_bonds()
                    parent, fragment = pairs[entry.frag_idx]
                    break
            else:
                raise Exception(
                    "Ligand not found: " + str(receptor) + " -- " + str(ligands)
                )

            sample = (receptor, parent, fragment)

            if self.transform:
                # Actually performs voxelization and fingerprinting.
                tmp = self.transform(sample)
                return tmp
            else:
                return sample

        # Only for debugging purposes
        except Exception as e:
            if entry is not None:
                print(f"\nMethod __getitem__ in 'fragment_dataset.py'. Error in pdb ID: {entry.pdb_id}; Ligand ID: {entry.ligand_id}\n {str(e)}", file=sys.stderr)
            else:
                print(f"\nMethod __getitem__ in 'fragment_dataset.py'.\n {str(e)}", file=sys.stderr)
            raise
