"""A Dataset that provides (receptor, parent, fragment) tuples by splitting
ligands on single bonds. Used in DeepFrag, for example.
"""

import argparse
from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Set, Optional, Any, Callable
from pathlib import Path
from torch.utils.data import Dataset
from collagen.core import args as user_args
from collagen.external.moad.split import full_moad_split
from ..cache_filter import CacheItemsToUpdate, load_cache_and_filter
import sys
from .... import Mol
from collagen.external.moad.interface import PairedPdbSdfCsvInterface
from collagen.core.molecules.mol import BackedMol


@dataclass
class MOADFragmentDataset_entry(object):

    """An entry in the MOADFragmentDataset."""

    pdb_id: str
    lig_to_frag_masses_chunk_idx: int
    ligand_id: str
    frag_idx: int


class MOADFragmentDataset(Dataset):

    """A Dataset that provides (receptor, parent, fragment) tuples by splitting
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
        """Initialize a MOADFragmentDataset.
        
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
            args (argparse.Namespace, optional): An optional set of arguments
                to control how the dataset is generated.

        Raises:
            ValueError: If the MOADInterface object is not initialized.
        """
        self.moad = moad
        self.split = split if split is not None else full_moad_split(moad)
        self.transform = transform
        self.args = args
        self.mol_props_param_validated = False
        self.smi_counts = {}
        self._index(cache_file, cache_cores)

    @staticmethod
    def add_fragment_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add user arguments so user can control how fragments are generated.

        Args:
            parent_parser (argparse.ArgumentParser): The parent parser to add
                arguments to.

        Returns:
            argparse.ArgumentParser: The parent parser with added arguments.
        """
        # TODO: Refactor how this is done with defaults, etc.

        # For many of these, good to define default values in args_defaults.py

        parser = parent_parser.add_argument_group("Fragment Dataset")

        parser.add_argument(
            "--min_frag_mass",
            required=False,
            type=float,
            # default=0,
            help="Consider only fragments with at least this molecular mass. Default is 0 Da.",
        )
        parser.add_argument(
            "--max_frag_mass",
            required=False,
            type=float,
            # default=150,
            help="Consider only fragments with at most this molecular mass. Default is 150 Da.",
        )
        parser.add_argument(
            "--max_frag_dist_to_recep",
            required=False,
            type=float,
            # default=4,
            help="Consider only fragments that have at least one atom that comes within this distance of any receptor atom. Default is 4 Å.",
        )
        parser.add_argument(
            "--min_frag_num_heavy_atoms",
            required=False,
            type=int,
            # default=1,
            help="Consider only fragments that have at least this number of heavy atoms. Default is 1.",
        )
        parser.add_argument(
            "--max_frag_num_heavy_atoms",
            required=False,
            type=int,
            # default=1,
            help="Consider only fragments that have at most this number of heavy atoms. Default is 1.",
        )
        parser.add_argument(
            "--mol_props",
            required=False,
            type=str,
            # default="",
            help='Consider only fragments that match selected chemical properties. A comma-separated list. Options are "aromatic", "aliphatic", "charged", "neutral". If specifying multiple properties (e.g., "aromatic,charged"), only fragments matching all properties (charged aromatics) will be considered. Default is "" (no filtering).',
        )
        parser.add_argument(
            "--max_frag_repeats",
            required=False,
            type=int,
            # default=None,
            help="If a given fragment has already been included in the dataset this many times, it will not be included again. This is to prevent common fragments (e.g., -OH) from dominating the datasets. If unspecified, no limit is imposed.",
        )

        return parent_parser

    def _lig_filter(
        self, args: argparse.Namespace, lig: "MOAD_ligand", lig_inf: Dict
    ) -> bool:
        """In the case of the fragment dataset, there is a filter applied to
        fragments, but not whole ligands. So everything passes. This is what is
        passed to cache_filter.load_cache_and_filter as the lig_filter_func
        parameter.
        
        Args:
            args (argparse.Namespace): The user arguments.
            lig (MOAD_ligand): The ligand to filter.
            lig_inf (Dict): The ligand's metadata.

        Returns:
            bool: Always True.
        """
        return True

    def _get_and_validate_mol_props_param(self, args: argparse.Namespace) -> List[str]:
        """Get the mol_props parameter from the user arguments and validates
        it.

        Args:
            args (argparse.Namespace): The user arguments.

        Raises:
            ValueError: If the mol_props parameter is invalid.

        Returns:
            List[str]: The mol_props parameter as a list of strings. E.g., 
                separates out string-list like "aromatic,charged" into
                `["aromatic", "charged"]`.
        """
        mol_props = args.mol_props.split(",")

        if not self.mol_props_param_validated:
            # This check to make sure mol_props validation is only done once.
            # Do some quick validation
            if "aromatic" in mol_props and "aliphatic" in mol_props:
                raise ValueError(
                    "Cannot specify both aromatic and aliphatic properties. They are mutually exclusive."
                )

            if "charged" in mol_props and "neutral" in mol_props:
                raise ValueError(
                    "Cannot specify both charged and neutral properties. They are mutually exclusive."
                )

            # If anything in molprops other than aromatic, aliphatic, charged,
            # or neutral is specified, raise an error saying which one is not
            # recognized.
            for prop in mol_props:
                if prop not in ["aromatic", "aliphatic", "charged", "neutral"]:
                    raise ValueError(f"Unrecognized property: {prop}")

            self.mol_props_param_validated = True

        return mol_props

    def _frag_filter(
        self,
        args: argparse.Namespace,
        mass: float,
        frag_dist_to_recep: float,
        frag_num_heavy_atom: int,
        frag_aromatic: bool,
        frag_charged: bool,
        frag_smi: str,
    ) -> bool:
        """Filter is passed to cache_filter.load_cache_and_filter via the
        make_dataset_entries_func parameter.

        Args:
            args (argparse.Namespace): The user arguments.
            mass (float): The fragment's molecular mass.
            frag_dist_to_recep (float): The fragment's distance to the receptor.
            frag_num_heavy_atom (int): The fragment's number of heavy atoms.
            frag_aromatic (bool): Whether the fragment is aromatic.
            frag_charged (bool): Whether the fragment is charged.
            frag_smi (str): The fragment's SMILES string.

        Returns:
            bool: Whether the fragment passes the filter.
        """
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
        
        if frag_num_heavy_atom > args.max_frag_num_heavy_atoms:
            if user_args.verbose:
                print(
                    f"Fragment rejected; has too many heavy atoms: {frag_num_heavy_atom}"
                )
            return False

        if args.max_frag_repeats is not None:
            if frag_smi not in self.smi_counts:
                self.smi_counts[frag_smi] = 0
            cnt = self.smi_counts[frag_smi]
            if cnt >= args.max_frag_repeats:
                if user_args.verbose:
                    print(f"Fragment rejected; already included {cnt} times.")
                return False
            self.smi_counts[frag_smi] += 1

        if args.mol_props != "":
            mol_props = self._get_and_validate_mol_props_param(args)

            if "aromatic" in mol_props and not frag_aromatic:
                if user_args.verbose:
                    print("Fragment rejected; not aromatic.")
                return False

            if "aliphatic" in mol_props and frag_aromatic:
                if user_args.verbose:
                    print("Fragment rejected; aromatic.")
                return False

            if "charged" in mol_props and not frag_charged:
                if user_args.verbose:
                    print("Fragment rejected; not charged.")
                return False

            if "neutral" in mol_props and frag_charged:
                if user_args.verbose:
                    print("Fragment rejected; charged.")
                return False

        return True

    def _make_dataset_entries_func(
        self, args: argparse.Namespace, pdb_id: str, lig_name: str, lig_inf: Dict
    ) -> List[MOADFragmentDataset_entry]:
        """Filter is passed to cache_filter.load_cache_and_filter as the
        make_dataset_entries_func parameter.

        Args:
            args (argparse.Namespace): The user arguments.
            pdb_id (str): The PDB ID of the ligand.
            lig_name (str): The name of the ligand.
            lig_inf (Dict): The ligand's information from the cache.
            
        Returns:
            List[MOADFragmentDataset_entry]: The list of entries to add to the
                dataset.
        """
        # Note that lig_inf contains all the data from the cache.

        # Here also doing some filtering of the fragments.
        frag_masses = lig_inf["frag_masses"]
        frag_dists_to_recep = lig_inf["frag_dists_to_recep"]
        frag_num_heavy_atoms = lig_inf["frag_num_heavy_atoms"]
        frag_aromatics = lig_inf["frag_aromatic"]
        frag_chargeds = lig_inf["frag_charged"]
        # Uses Chem.MolToSmiles, so should be cannonical
        frag_smiles = lig_inf["frag_smiles"]

        entries_to_return = []
        for frag_idx in range(len(frag_masses)):
            mass = frag_masses[frag_idx]
            dist_to_recep = 0 if len(frag_dists_to_recep) == 0 else frag_dists_to_recep[frag_idx]
            num_heavy_atom = frag_num_heavy_atoms[frag_idx]
            frag_aromatic = frag_aromatics[frag_idx]
            frag_charged = frag_chargeds[frag_idx]
            frag_smi = frag_smiles[frag_idx]

            if self._frag_filter(
                args,
                mass,
                dist_to_recep,
                num_heavy_atom,
                frag_aromatic,
                frag_charged,
                frag_smi,
            ):
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
        """Create the cache and filtered cache, here referred to as an index.

        Args:
            cache_file (Optional[Union[str, Path]], optional): The path to the
                cache file. Defaults to None.
            cores (int, optional): The number of cores to use. Defaults to 1.
        """
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
                frag_aromatic=True,
                frag_charged=True,
            ),
            cache_file,
            cores,
        )

        self._ligand_index_cached = cache
        self._internal_index_valids_filtered = filtered_cache

    def __len__(self) -> int:
        """Return the number of entries in the dataset.
        
        Returns:
            int: The number of entries in the dataset.
        """
        return len(self._internal_index_valids_filtered)

    def __getitem__(self, idx: int) -> Tuple[Mol, Mol, Mol]:
        """Return (receptor, parent, fragment)
        
        Args:
            idx (int): The index of the entry to return.
            
        Returns:
            Tuple[Mol, Mol, Mol]: (receptor, parent, fragment)
        """
        assert 0 <= idx < len(self), "Index out of bounds"
        entry = None
        counter = 1
        max_counter = 3
        # For some reason, id we repeat this code more than 1 times, then we
        # avoid 'a ligand not found' or 'atoms do not have coordinates'
        # exceptions Is this only for Windows platforms due to the
        # multiprocessing implemented in the loader.py file? I don't know Can
        # this occur on Unix-type platforms with the fork multiprocessing
        # implemented in the loader.py file? I don't know, if required to
        # experiment
        while counter <= max_counter:
            try:
                entry = self._internal_index_valids_filtered[idx]

                receptor, ligands = self.moad[entry.pdb_id][entry.lig_to_frag_masses_chunk_idx]
                assert isinstance(receptor, BackedMol), "Receptor not found"
                assert len(ligands) >= 1, "Ligand list not found"

                # with open("/mnt/extra/fragz2.txt", "a") as f:
                #     f.write(receptor.meta["name"] + "\t" + str(ligands) + "\n")

                # This chunk has many ligands. You need to look up the one that matches
                # entry.ligand_id (the one that actually corresponds to this entry).
                # Once you find it, actually do the fragmenting.
                for ligand in ligands:
                    if ligand.meta["moad_ligand"].name == entry.ligand_id:
                        if isinstance(self.moad, PairedPdbSdfCsvInterface):
                            list_frag_and_act = self.moad.frag_and_act_x_parent_x_sdf_x_pdb[entry.ligand_id]
                            assert 0 <= entry.frag_idx < len(list_frag_and_act), "Fragment index out of bounds"

                            backed_frag = list_frag_and_act[entry.frag_idx][2]
                            parent = ligand.meta["moad_ligand"].backed_parent  # BackedMol(rdmol=ligand.rdmol)
                            fragment = backed_frag  # BackedMol(rdmol=backed_frag.rdmol)
                            break
                        else:
                            pairs = ligand.split_bonds()
                            parent, fragment = pairs[entry.frag_idx]
                            break
                else:
                    raise Exception(f"Ligand not found: {str(receptor)} -- {str(ligands)}")

                assert isinstance(parent, BackedMol), "Parent not found"
                assert isinstance(fragment, BackedMol), "Fragment not found"
                sample = (receptor, parent, fragment, entry.ligand_id, entry.frag_idx)
                assert len(sample) == 5, "Sample size is different to 5"

                # Actually performs voxelization and fingerprinting.
                return self.transform(sample) if self.transform else sample

            except AssertionError as e:
                print(
                    f"\nMethod __getitem__ in 'fragment_dataset.py'. Assertion Error: {str(e)}",
                    file=sys.stderr,
                )
                raise e
            except Exception as e:
                if entry is not None:
                    print(
                        f"\nMethod __getitem__ in 'fragment_dataset.py'. Error in pdb ID: {entry.pdb_id}; Ligand ID: {entry.ligand_id}\n {counter} times: {str(e)}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"\nMethod __getitem__ in 'fragment_dataset.py'.\n {counter} times: {str(e)}",
                        file=sys.stderr,
                    )
                    counter = max_counter

                if counter == max_counter:
                    raise
                else:
                    counter += 1
