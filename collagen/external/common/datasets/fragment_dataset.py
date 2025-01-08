"""A Dataset that provides (receptor, parent, fragment) tuples by splitting
ligands on single bonds. Used in DeepFrag, for example.
"""

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Dict, Union, Tuple, Optional, Any, Callable
from pathlib import Path
from collagen.external.common.parent_targets_ligands import Parent_ligand
from collagen.external.common.types import StructuresSplit
from collagen.external.common.cache_filter import CacheItemsToUpdate, load_cache_and_filter
from collagen.external.paired_csv.interface import PairedCsvInterface
from torch.utils.data import Dataset  # type: ignore
from collagen.core import args as user_args
from collagen.external.common.split import create_full_dataset_as_single_split
from collagen.core.molecules.mol import Mol
import sys
from collagen.core.molecules.mol import BackedMol
import torch
import numpy as np
import os


if TYPE_CHECKING:
    from collagen.external.common.parent_interface import ParentInterface
    from collagen.core.molecules.mol import Mol


@dataclass
class FragmentDataset_entry(object):

    """An entry in the FragmentDataset."""

    pdb_id: str
    lig_to_frag_masses_chunk_idx: int
    ligand_id: str
    frag_idx: int


class FragmentDataset(Dataset):

    """A Dataset that provides (receptor, parent, fragment) tuples by splitting
    ligands on single bonds. Used in DeepFrag, for example.

    Args:
        data_interface (ParentInterface): An initialized ParentInterface object.
        cache_file (str, optional): Path to a cache file to store or load
            fragment metadata.
        cache_cores (int, optional): If a cache file is not found, use this
            many cores to compute a new cache.
        split (StructuresSplit, optional): An optional split to constrain the space
            of examples.
        transform (Callable[[Mol, Mol, Mol], Any], optional): An optional
            transformation function to invoke before returning samples. Takes
            the arguments (receptor, parent, fragment) as Mol objects.
    """

    data_interface: "ParentInterface"
    split: "StructuresSplit"

    # function that performs voxelization and fingerprinting
    # Note that transform accepts a tuple or list: Tuple[Mol, Mol, Mol, str, int]
    transform: Optional[Callable[[Tuple[Mol, Mol, Mol, str, int]], Any]]

    # The internal listing of every valid fragment example. This index is
    # generated on each run based on the runtime filters: (targets, smiles,
    # fragment_size).
    _internal_index_valids_filtered: List[FragmentDataset_entry]

    def __init__(
        self,
        data_interface: "ParentInterface",
        cache_file: Optional[Union[str, Path]] = None,
        cache_cores: int = 1,
        split: Optional["StructuresSplit"] = None,
        transform: Optional[Callable[[Tuple[Mol, Mol, Mol, str, int]], Any]] = None,
        args: Optional[argparse.Namespace] = None,
    ):
        """Initialize a FragmentDataset.

        Args:
            data_interface (ParentInterface): An initialized ParentInterface object.
            cache_file (str, optional): Path to a cache file to store or load
                fragment metadata.
            cache_cores (int, optional): If a cache file is not found, use this
                many cores to compute a new cache.
            split (StructuresSplit, optional): An optional split to constrain the space
                of examples.
            transform (Callable[[Mol, Mol, Mol, str, int], Any], optional): An
                optional transformation function to invoke before returning
                samples. Takes the arguments (receptor, parent, fragment) as
                Mol objects.
            args (argparse.Namespace, optional): An optional set of arguments
                to control how the dataset is generated.

        Raises:
            ValueError: If the MOADInterface object is not initialized.
        """
        self.data_interface = data_interface
        self.split = split if split is not None else create_full_dataset_as_single_split(data_interface)
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
            help='Consider only fragments that match selected chemical properties. A comma-separated list. Options are "aromatic", "aliphatic", "acid", "base", "neutral". If specifying multiple properties (e.g., "aromatic,acid"), only fragments matching all properties (acidic aromatics) will be considered. Default is "" (no filtering).',
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
        self, args: argparse.Namespace, lig: "Parent_ligand", lig_inf: Dict
    ) -> bool:
        """In the case of the fragment dataset, there is a filter applied to
        fragments, but not whole ligands. So everything passes. This is what is
        passed to cache_filter.load_cache_and_filter as the lig_filter_func
        parameter.

        Args:
            args (argparse.Namespace): The user arguments.
            lig (Parent_ligand): The ligand to filter.
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
                separates out string-list like "aromatic,acid" into
                `["aromatic", "acid"]`.
        """
        mol_props = args.mol_props.split(",")

        if not self.mol_props_param_validated:
            # This check to make sure mol_props validation is only done once.
            # Do some quick validation
            if "aromatic" in mol_props and "aliphatic" in mol_props:
                raise ValueError(
                    "Cannot specify both aromatic and aliphatic properties. They are mutually exclusive."
                )

            if "acid" in mol_props and "neutral" in mol_props:
                raise ValueError(
                    "Cannot specify both acid and neutral properties. They are mutually exclusive."
                )

            if "base" in mol_props and "neutral" in mol_props:
                raise ValueError(
                    "Cannot specify both base and neutral properties. They are mutually exclusive."
                )

            # If anything in molprops other than aromatic, aliphatic, acid,
            # base, or neutral is specified, raise an error saying which one is
            # not recognized.
            for prop in mol_props:
                if prop not in ["aromatic", "aliphatic", "acid", "base", "neutral"]:
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
        frag_acid: bool,
        frag_base: bool,
        frag_neutral: bool,
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
            frag_acid (bool): Whether the fragment is acid.
            frag_base (bool): Whether the fragment is base.
            frag_neutral (bool): Whether the fragment is neutral.
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

            if "acid" in mol_props and not frag_acid:
                if user_args.verbose:
                    print("Fragment rejected; not acid.")
                return False

            if "base" in mol_props and not frag_base:
                if user_args.verbose:
                    print("Fragment rejected; not base.")
                return False

            if "neutral" in mol_props and not frag_neutral:
                if user_args.verbose:
                    print("Fragment rejected; not neutral.")
                return False

        return True

    def _make_dataset_entries_func(
        self, args: argparse.Namespace, pdb_id: str, lig_name: str, lig_inf: Dict
    ) -> List[FragmentDataset_entry]:
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
        frag_acids = lig_inf["frag_acid"]
        frag_bases = lig_inf["frag_base"]
        frag_neutrals = lig_inf["frag_neutral"]
        # Uses Chem.MolToSmiles, so should be cannonical
        frag_smiles = lig_inf["frag_smiles"]

        entries_to_return = []
        for frag_idx in range(len(frag_masses)):
            mass = frag_masses[frag_idx]
            dist_to_recep = (
                0 if len(frag_dists_to_recep) == 0 else frag_dists_to_recep[frag_idx]
            )
            num_heavy_atom = frag_num_heavy_atoms[frag_idx]
            frag_aromatic = frag_aromatics[frag_idx]
            frag_acid = frag_acids[frag_idx]
            frag_base = frag_bases[frag_idx]
            frag_neutral = frag_neutrals[frag_idx]
            frag_smi = frag_smiles[frag_idx]

            if self._frag_filter(
                args,
                mass,
                dist_to_recep,
                num_heavy_atom,
                frag_aromatic,
                frag_acid,
                frag_base,
                frag_neutral,
                frag_smi,
            ):
                entries_to_return.append(
                    FragmentDataset_entry(
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
        # Make sure self.args is not None
        assert self.args is not None, "MOADFragmentDataset args is None"

        cache, filtered_cache = load_cache_and_filter(
            self._lig_filter,
            self.data_interface,
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
                frag_acid=True,
                frag_base=True,
                frag_neutral=True,
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

    def __getitem__(self, idx: int) -> Union[None, Tuple[Mol, Mol, Mol, str, int], Any]:
        """Return (receptor, parent, fragment)

        Args:
            idx (int): The index of the entry to return.

        Returns:
            Union[None, Tuple[Mol, Mol, Mol, str, int], Any]: (receptor,
            parent, fragment, ligand_id, frag_idx)
        """
        assert 0 <= idx < len(self), "Index out of bounds"

        entry: Union[FragmentDataset_entry, None] = None
        counter = 1
        max_counter = 3

        while counter <= max_counter:
            try:
                entry = self._internal_index_valids_filtered[idx]
                receptor, ligands = self.data_interface[entry.pdb_id][
                    entry.lig_to_frag_masses_chunk_idx
                ]

                assert isinstance(receptor, BackedMol), "Receptor not found"
                assert len(ligands) >= 1, "Ligand list not found"

                for ligand in ligands:
                    if ligand.meta["moad_ligand"].name == entry.ligand_id:
                        if isinstance(self.data_interface, PairedCsvInterface):
                            list_frag_and_act = (
                                self.data_interface.frag_and_act_x_parent_x_sdf_x_pdb[
                                    entry.ligand_id
                                ]
                            )
                            assert (
                                0 <= entry.frag_idx < len(list_frag_and_act)
                            ), "Fragment index out of bounds"

                            backed_frag = list_frag_and_act[entry.frag_idx][2]
                            parent = ligand.meta[
                                "moad_ligand"
                            ].backed_parent
                            fragment = backed_frag
                            break
                        else:
                            pairs = ligand.split_bonds()
                            parent, fragment = pairs[entry.frag_idx]
                            break
                else:
                    raise Exception(
                        f"Ligand not found: {str(receptor)} -- {str(ligands)}"
                    )

                assert isinstance(parent, BackedMol), "Parent not found"
                assert isinstance(fragment, BackedMol), "Fragment not found"
                sample = (receptor, parent, fragment, entry.ligand_id, entry.frag_idx)
                assert len(sample) == 5, "Sample size is not 5"

                resp = self.transform(sample) if self.transform else sample

                # Add debug visualization for first sample
                # print(idx, hasattr(self, '_debug_saved'))
                # if idx == 0 and not hasattr(self, '_debug_saved'):
                if not os.path.exists("debug_viz"):
                    print("HEREEEEEEE!!!!!!!!!!")
                    # self._debug_saved = True
                    self._save_debug_visualization(sample, fragment.connectors[0])
                    sys.exit(0)  # Debugging, so stop immediately.

                return resp

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

    def _save_debug_visualization(self, sample: Tuple[Mol, Mol, Mol, str, int], center: np.ndarray):
        """Save debug visualization of voxelization.
        
        Args:
            sample: (receptor, parent, fragment, ligand_id, frag_idx)
            center: Center point for voxelization
        """
        # Create debug directory
        os.makedirs("debug_viz", exist_ok=True)
        
        receptor, parent, fragment, ligand_id, frag_idx = sample
        
        # Save the PDB file before voxelization
        with open(f"debug_viz/receptor.pdb", "w") as f:
            f.write(receptor.pdb())
        
        # Get voxel params from parent class
        print("1")
        voxel_params = self.transform.keywords['voxel_params']
        
        print("2")

        # Get the voxelized grid using the same voxelization as transform
        tensor = receptor.voxelize(voxel_params, cpu=True, center=center)

        print("3")
        
        # Convert tensor to numpy
        voxel = tensor.numpy()

        print("4", voxel)
        
        # Get grid parameters
        nx = ny = nz = voxel.shape[2]  # Assuming cubic grid 
        spacing = voxel_params.resolution
        origin = -(nx * spacing) / 2.0  # Center grid at origin
        
        print("TEST", voxel.shape)
        # Save each channel as DX file
        for channel in range(voxel.shape[1]):
            grid_data = voxel[0, channel]  # First batch, each channel
            with open(f"debug_viz/channel_{channel}.dx", "w") as f:
                f.write("object 1 class gridpositions counts %d %d %d\n" % (nx, ny, nz))
                f.write("origin %.2f %.2f %.2f\n" % (origin, origin, origin))
                f.write("delta %.2f 0.0 0.0\n" % spacing)
                f.write("delta 0.0 %.2f 0.0\n" % spacing)
                f.write("delta 0.0 0.0 %.2f\n" % spacing)
                f.write("object 2 class gridconnections counts %d %d %d\n" % (nx, ny, nz))
                f.write("object 3 class array type double rank 0 items %d data follows\n" % (nx*ny*nz))
                
                # Write grid data
                count = 0
                for val in grid_data.flatten():
                    f.write("%.6f " % val)
                    count += 1
                    if count % 3 == 0:
                        f.write("\n")
                if count % 3 != 0:
                    f.write("\n")
        
        # Save center point
        with open(f"debug_viz/center.txt", "w") as f:
            f.write(f"x,y,z\n")
            f.write(f"{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}\n")