"""Calculating molecular properties for the many ligands and fragments in the
BindingMOAD database is expensive. These classes cache the calculations for
quick look-up later.
"""

import argparse
from dataclasses import dataclass
import json
from collagen.external.common.chem_props import (
    is_acid,
    is_aromatic,
    is_base,
    is_neutral,
)
from collagen.external.common.parent_interface import ParentInterface
from collagen.external.common.parent_targets_ligands import Parent_target
from collagen.external.common.types import StructuresSplit
from collagen.external.paired_csv.targets_ligands import PairedCsv_ligand
from torch import multiprocessing  # type: ignore
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Tuple, Union, Any, Optional, List
from tqdm.std import tqdm  # type: ignore
import numpy as np  # type: ignore
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles  # type: ignore
from scipy.spatial.distance import cdist
from .utils import fix_smiles

@dataclass
class CacheItemsToUpdate(object):

    """Dataclass describing which calculated properties should be added to the
    cache.
    """

    # Updatable
    lig_mass: bool = False
    murcko_scaffold: bool = False
    num_heavy_atoms: bool = False
    frag_masses: bool = False
    frag_num_heavy_atoms: bool = False
    frag_dists_to_recep: bool = False
    frag_smiles: bool = False  # str || bool
    frag_aromatic: bool = False
    frag_acid: bool = False
    frag_base: bool = False
    frag_neutral: bool = False

    def updatable(self) -> bool:
        """Return True if any of the properties are updatable.

        Returns:
            bool: True if any of the properties are updatable.
        """
        # Updatable
        return any(
            [
                self.lig_mass,
                self.murcko_scaffold,
                self.num_heavy_atoms,
                self.frag_masses,
                self.frag_dists_to_recep,
                self.frag_smiles,
                self.frag_num_heavy_atoms,
                self.frag_aromatic,
                self.frag_acid,
                self.frag_base,
                self.frag_neutral,
            ]
        )


# MOAD_REF: "MOADInterface"
CACHE_ITEMS_TO_UPDATE = CacheItemsToUpdate()


def _set_molecular_prop(func: Callable, func_input: Any, default_if_error: Any) -> Any:
    """Provide a way to provide a default value if function fails.

    Args:
        func (callable): The function to call.
        func_input (Any): The input to the function.
        default_if_error (Any): The default value to return if the function
            fails.

    Returns:
        Any: The result of the function, or the default value if the function
            fails.
    """
    try:
        return func(func_input)
    except Exception as e:
        # Save the exception to err.txt
        # with open("err.txt", "a") as f:
        #     f.write(f"{func_input} failed to calculate {func.__name__}\n")
        #     f.write(str(e) + "\n")

        return default_if_error


def get_info_given_pdb_id(payload: Tuple[str, Parent_target, CacheItemsToUpdate]) -> Tuple[str, dict]:
    """Given a PDB ID, looks up the PDB in BindingMOAD, and calculates the
    molecular properties specified in CACHE_ITEMS_TO_UPDATE. Returns a tuple
    with the pdb id and a dictionary with the associated information.

    Args:
        payload (Tuple[str, Parent_target, CacheItemsToUpdate]): A tuple
            containing the PDB ID, the ParentInterface object, and the
            CacheItemsToUpdate object.

    Returns:
        Tuple[str, dict]: A tuple containing the PDB ID and a dictionary with
            the associated information.
    """
    # global MOAD_REF
    # global CACHE_ITEMS_TO_UPDATE

    # moad_entry_info = moad[pdb_id]
    pdb_id = payload[0]
    target = payload[1]
    cache_items_to_update = payload[2]

    # Maps string to dict
    lig_infs = {}
    for lig_chunk_idx in range(len(target)):
        try:
            # Unpack info to get ligands
            receptor, ligands = target[lig_chunk_idx]
        except Exception:
            # Note that prody can't parse some PDBs for some reason. Examples:
            # 1vif
            continue

        if len(ligands) == 0:
            # Strangely, some entries in Binding MOAD don't actually have
            # non-protein/peptide ligands. For example: 2awu 2yno 5n70
            continue

        for lig in ligands:
            lig_name = lig.meta["moad_ligand"].name
            lig_infs[lig_name] = {"lig_chunk_idx": lig_chunk_idx}

            # First, deal with properties that apply to the entire ligand (not
            # each fragment)

            # Updatable
            if cache_items_to_update.lig_mass:
                lig_infs[lig_name]["lig_mass"] = _set_molecular_prop(
                    lambda x: x.mass, lig, 999999
                )

            if cache_items_to_update.murcko_scaffold:
                try:
                    smi = lig.smiles(True)
                    assert smi is not None, "SMILES is None"
                    smi_fixed = fix_smiles(smi)
                    scaffold_smi = MurckoScaffoldSmilesFromSmiles(
                        smi_fixed, includeChirality=True
                    )
                    lig_infs[lig_name]["murcko_scaffold"] = scaffold_smi
                except Exception:
                    lig_infs[lig_name]["murcko_scaffold"] = ""

            if cache_items_to_update.num_heavy_atoms:
                lig_infs[lig_name]["num_heavy_atoms"] = _set_molecular_prop(
                    lambda x: x.num_heavy_atoms, lig, 999999
                )

            # Now deal with properties by fragment (not entire ligand)

            # Prevents frags from being unbound.
            frags = []

            if (
                cache_items_to_update.frag_masses
                or cache_items_to_update.frag_num_heavy_atoms
                or cache_items_to_update.frag_dists_to_recep
                or cache_items_to_update.frag_smiles
                or cache_items_to_update.frag_aromatic
                or cache_items_to_update.frag_acid
                or cache_items_to_update.frag_base
                or cache_items_to_update.frag_neutral
            ):
                moad_ligand_ = lig.meta["moad_ligand"]
                if isinstance(moad_ligand_, PairedCsv_ligand):
                    # Get all the fragments from an additional csv file
                    frags = []
                    for _, _, backed_frag, _, _ in moad_ligand_.fragment_and_act:
                        frags.append([moad_ligand_.smiles, backed_frag])
                else:
                    # Get all the fragments
                    frags = _set_molecular_prop(lambda x: x.split_bonds(), lig, [])

            if cache_items_to_update.frag_masses:
                lig_infs[lig_name]["frag_masses"] = _set_molecular_prop(
                    lambda f: [x[1].mass for x in f], frags, []
                )

            if cache_items_to_update.frag_num_heavy_atoms:
                lig_infs[lig_name]["frag_num_heavy_atoms"] = _set_molecular_prop(
                    lambda f: [x[1].num_heavy_atoms for x in f], frags, []
                )

            if cache_items_to_update.frag_dists_to_recep:
                lig_infs[lig_name]["frag_dists_to_recep"] = _set_molecular_prop(
                    lambda f: [np.min(cdist(x[1].coords, receptor.coords)) for x in f],
                    frags,
                    [],
                )

            if cache_items_to_update.frag_smiles:
                # Helpful for debugging, mostly.
                lig_infs[lig_name]["frag_smiles"] = _set_molecular_prop(
                    lambda f: [x[1].smiles(True) for x in f],
                    frags,
                    [],
                )

            if cache_items_to_update.frag_aromatic:
                lig_infs[lig_name]["frag_aromatic"] = _set_molecular_prop(
                    lambda f: [is_aromatic(x[1].rdmol) for x in f], frags, []
                )

            if cache_items_to_update.frag_acid:
                lig_infs[lig_name]["frag_acid"] = _set_molecular_prop(
                    lambda f: [is_acid(x[1].rdmol) for x in f], frags, []
                )

            if cache_items_to_update.frag_base:
                lig_infs[lig_name]["frag_base"] = _set_molecular_prop(
                    lambda f: [is_base(x[1].rdmol) for x in f], frags, []
                )

            if cache_items_to_update.frag_neutral:
                lig_infs[lig_name]["frag_neutral"] = _set_molecular_prop(
                    lambda f: [is_neutral(x[1].rdmol) for x in f], frags, []
                )

    return pdb_id, lig_infs


def _set_cache_params_to_update(cache: Dict[str, Dict[str, Dict[str, Any]]]):
    """Look at the current cache and determines which properties need to be
    added to it. Sets flags in CACHE_ITEMS_TO_UPDATE as appropriate.

    Args:
        cache (dict): The current cache.
    """
    pdb_ids_in_cache: List[str] = list(cache.keys())
    if not pdb_ids_in_cache:  # empty
        return

    # Find the first entry in the existing cache that is not empty. Need to do
    # this check because sometimes cache entries are {} (not able to extract
    # ligand, for example).
    assert len(pdb_ids_in_cache) > 0, "Cache is empty. Something is wrong."

    pdb_id_in_cache = ""
    for pdb_id_in_cache in pdb_ids_in_cache:
        if cache[pdb_id_in_cache] != {}:
            break
    ref_pdb_inf = cache[pdb_id_in_cache]

    if len(ref_pdb_inf.keys()) == 0:
        return

    first_lig = ref_pdb_inf[list(ref_pdb_inf.keys())[0]]

    # Updatable
    global CACHE_ITEMS_TO_UPDATE
    if "lig_mass" in first_lig:
        CACHE_ITEMS_TO_UPDATE.lig_mass = False
    if "frag_masses" in first_lig:
        CACHE_ITEMS_TO_UPDATE.frag_masses = False
    if "murcko_scaffold" in first_lig:
        CACHE_ITEMS_TO_UPDATE.murcko_scaffold = False
    if "num_heavy_atoms" in first_lig:
        CACHE_ITEMS_TO_UPDATE.num_heavy_atoms = False
    if "frag_num_heavy_atoms" in first_lig:
        CACHE_ITEMS_TO_UPDATE.frag_num_heavy_atoms = False
    if "frag_dists_to_recep" in first_lig:
        CACHE_ITEMS_TO_UPDATE.frag_dists_to_recep = False
    if "frag_smiles" in first_lig:
        CACHE_ITEMS_TO_UPDATE.frag_smiles = False
    if "frag_aromatic" in first_lig:
        CACHE_ITEMS_TO_UPDATE.frag_aromatic = False
    if "frag_acid" in first_lig:
        CACHE_ITEMS_TO_UPDATE.frag_acid = False
    if "frag_base" in first_lig:
        CACHE_ITEMS_TO_UPDATE.frag_base = False
    if "frag_neutral" in first_lig:
        CACHE_ITEMS_TO_UPDATE.frag_neutral = False


def _build_cache_file(
    filename: Optional[str],
    data_interface: "ParentInterface",
    cache_items_to_update: CacheItemsToUpdate,
    cores: Optional[int] = None,
) -> dict:
    """Builds/updates the whole BindingMOAD cache (on disk).

    Args:
        filename (str): The filename to save the cache to.
        data_interface (ParentInterface): The ParentInterface object to use.
        cache_items_to_update (CacheItemsToUpdate): The cache items to update.
        cores (int, optional): The number of cores to use. Defaults to None.

    Returns:
        dict: The cache.
    """
    # Load existing cache if it exists. So you can add to it.
    if filename and os.path.exists(filename):
        with open(filename) as f:
            cache: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)
    else:
        # No existing cache, so start empty.
        cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Setup and modify cache items to update.
    global CACHE_ITEMS_TO_UPDATE
    CACHE_ITEMS_TO_UPDATE = cache_items_to_update

    _set_cache_params_to_update(cache)

    if not CACHE_ITEMS_TO_UPDATE.updatable():
        # Nothing to update
        return cache

    # global MOAD_REF
    # MOAD_REF = moad

    pdb_ids_queue = data_interface.targets
    list_ids_moad: List[Tuple[str, Parent_target, CacheItemsToUpdate]] = [
        (pdb_id, data_interface[pdb_id], CACHE_ITEMS_TO_UPDATE) for pdb_id in data_interface.targets
    ]
    # NOTE: Filename specified via --cache parameter

    print("Building/updating " + (filename or "dataset"))
    with multiprocessing.Pool(cores) as p:
        for pdb_id, lig_infs in tqdm(
            p.imap_unordered(get_info_given_pdb_id, list_ids_moad),
            total=len(pdb_ids_queue),
            desc="Building cache",
        ):
            pdb_id = pdb_id.lower()

            print(pdb_id)  # debugging

            if pdb_id not in cache:
                cache[pdb_id] = {}

            for lig_name in lig_infs.keys():
                if lig_name not in cache[pdb_id]:
                    cache[pdb_id][lig_name] = {}
                lig_inf = lig_infs[lig_name]
                for prop_name in lig_inf.keys():
                    prop_val = lig_inf[prop_name]
                    cache[pdb_id][lig_name][prop_name] = prop_val
        p.close()

    # Save cache with updated information.
    if filename:
        with open(filename, "w") as f:
            json.dump(cache, f, indent=4)

    return cache


def load_cache_and_filter(
    lig_filter_func: Callable,  # The function used to filter the ligands
    data_interface: "ParentInterface",
    split: "StructuresSplit",
    args: argparse.Namespace,
    make_dataset_entries_func: Callable,
    cache_items_to_update: CacheItemsToUpdate,
    cache_file: Optional[Union[str, Path]] = None,
    cores: int = 1,
) -> Tuple[dict, list]:
    """Not only builds/gets the cache, but also filters the properties to
    retain only those BindingMOAD entries for training. For example, could
    filter by ligand mass.

    Args:
        lig_filter_func (function): The function used to filter the ligands.
        data_interface (ParentInterface): The ParentInterface object to use.
        split (StructuresSplit): The StructuresSplit object to use.
        args (argparse.Namespace): The arguments.
        make_dataset_entries_func (function): The function to make the dataset
            entries.
        cache_items_to_update (CacheItemsToUpdate): The cache items to update.
        cache_file (Optional[Union[str, Path]], optional): The cache file to
            use. Defaults to None.
        cores (int, optional): The number of cores to use. Defaults to 1.

    Returns:
        Tuple[dict, list]: The cache and the filtered cache.
    """
    # This function gets called from the dataset (e.g., fragment_dataset.py),
    # where the actual filters are set (via lig_filter_func).

    # Returns tuple, cache and filtered_cache.

    cache_file = Path(cache_file) if cache_file is not None else None
    cache = _build_cache_file(
        str(cache_file) if cache_file else None,
        data_interface,
        cache_items_to_update,
        cores=cores,
    )

    total_complexes = 0
    total_complexes_with_both_in_split = 0
    total_complexes_passed_lig_filter = 0
    total_complexes_with_useful_fragments = 0
    pdbs_with_useful_fragments = set()

    filtered_cache = []
    for pdb_id in tqdm(split.targets, desc="Runtime filters"):
        pdb_id = pdb_id.lower()

        # If the PDB ID is not in the cache, throw an error. Cache probably
        # corrupt.
        if pdb_id not in cache.keys():
            # missing entry in cache. Has probably become corrupted.
            raise Exception(
                f"Entry {pdb_id} is not present in the index. Has the cache file been corrupted? Try moving/deleting {str(cache_file)} and rerunning to regenerate the cache."
            )

        # Iterate through receptor, ligand info.
        receptor_inf = cache[pdb_id]
        for lig_name in receptor_inf.keys():
            lig_inf = receptor_inf[lig_name]

            # Enforce whole-ligand filters and not-in-same-split filters.
            fails_filter = False
            # Search for ligand_name.
            for lig in data_interface[pdb_id].ligands:
                if lig.name != lig_name:
                    # Not the ligand you're looking for. Continue searching.
                    continue

                # You've found the ligand.

                total_complexes += 1

                if lig.smiles not in split.smiles:
                    # It is not in the split, so always skip it.
                    print(
                        f"Skipping {pdb_id}:{lig_name} because ligand not allowed in this split to ensure independence."
                    )
                    fails_filter = True
                    break

                total_complexes_with_both_in_split += 1

                if not lig_filter_func(args, lig, lig_inf):
                    # You've found the ligand, but it doesn't pass the filter.
                    # (Note that lig_filter_func likely just returns true, so
                    # code never gets here, everything passes).
                    print(
                        f"Skipping {pdb_id}:{lig_name} because ligand did not pass whole-ligand filter."
                    )
                    fails_filter = True
                    break

                total_complexes_passed_lig_filter += 1

            if fails_filter:
                continue

            # Add to filtered cache.
            examples_to_add = make_dataset_entries_func(args, pdb_id, lig_name, lig_inf)
            filtered_cache.extend(examples_to_add)
            if len(examples_to_add) == 0:
                print(
                    f"Skipping {pdb_id}:{lig_name} because no valid fragments for ligand found."
                )
            else:
                total_complexes_with_useful_fragments += 1
                pdbs_with_useful_fragments.add(pdb_id)

    if not filtered_cache:
        raise Exception(
            "No ligands passed the filters. Could be that filters are too strict, or perhaps there is a problem with your CSV file. Consider using `--verbose True` to debug."
        )

    print(f"\nSPLIT SUMMARY AFTER FRAGMENTING/FILTERING: {split.name}")
    print(f"Proteins (some with multiple ligands): {len(split.targets)}")
    print(f"Unique protein/ligand complexes: {total_complexes}")
    print(
        f"Complexes with both receptor and ligand in this split: {total_complexes_with_both_in_split}"
    )
    print(
        f"Complexes that also passed whole-ligand filter: {total_complexes_passed_lig_filter}"
    )
    print(
        f"Complexes that also had useful fragments: {total_complexes_with_useful_fragments}"
    )
    print(f"Protein/parent/fragment examples: {len(filtered_cache)}")
    print(
        f"Proteins with useful fragments (any ligand): {len(pdbs_with_useful_fragments)}"
    )
    print("")

    return cache, filtered_cache
