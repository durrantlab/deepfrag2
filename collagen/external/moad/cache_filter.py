import argparse
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
from typing import Tuple, Union, Any, Optional
from tqdm.std import tqdm
import numpy as np
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
from scipy.spatial.distance import cdist
from collagen.core.molecules.smiles_utils import standardize_smiles
from .moad_utils import fix_moad_smiles

# Calculating molecular properties for the many ligands and fragments in the
# BindingMOAD database is expensive. These classes cache the calculations for
# quick look-up later.


@dataclass
class CacheItemsToUpdate(object):
    # Dataclass describing which calculated properties should be added to the
    # cache.

    # Updatable
    lig_mass: bool = False
    murcko_scaffold: bool = False
    num_heavy_atoms: bool = False
    frag_masses: bool = False
    frag_num_heavy_atoms: bool = False
    frag_dists_to_recep: bool = False
    frag_smiles: str = False  # str || bool

    def updatable(self) -> bool:
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
            ]
        )


# MOAD_REF: "MOADInterface"
CACHE_ITEMS_TO_UPDATE = CacheItemsToUpdate()


def _set_molecular_prop(func, func_input, default_if_error):
    # Provides a way to provide a default value if function fails.
    try:
        return func(func_input)
    except Exception:
        return default_if_error


def _get_info_given_pdb_id(pdb_id: str, moad_entry_info, cache_items_to_update) -> Tuple[str, dict]:
    # Given a PDB ID, looks up the PDB in BindingMOAD, and calculates the
    # molecular properties specified in CACHE_ITEMS_TO_UPDATE. Returns a tuple
    # with the pdb id and a dictionary with the associated information.

    # global MOAD_REF
    # global CACHE_ITEMS_TO_UPDATE

    # moad_entry_info = moad[pdb_id]

    lig_infs = {}
    for lig_chunk_idx in range(len(moad_entry_info)):
        try:
            # Unpack info to get ligands
            receptor, ligands = moad_entry_info[lig_chunk_idx]
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

            # Updatable
            if cache_items_to_update.lig_mass:
                lig_infs[lig_name]["lig_mass"] = _set_molecular_prop(
                    lambda x: x.mass, lig, 999999
                )

            if cache_items_to_update.murcko_scaffold:
                try:
                    smi_fixed = fix_moad_smiles(lig.smiles(True))
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

            if (
                    cache_items_to_update.frag_masses
                    or cache_items_to_update.frag_num_heavy_atoms
                    or cache_items_to_update.frag_dists_to_recep
                    or cache_items_to_update.frag_smiles
            ):
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
                        lambda f: [
                            np.min(cdist(x[1].coords, receptor.coords)) for x in f
                        ],
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

    return pdb_id, lig_infs


def _set_cache_params_to_update(cache):
    # This looks at the current cache and determines which properties need to be
    # added to it. Sets flags in CACHE_ITEMS_TO_UPDATE as appropriate.

    pdb_ids_in_cache = list(cache.keys())
    if not pdb_ids_in_cache:  # empty
        return

    # Find the first entry in the existing cache that is not empty. Need to do
    # this check because sometimes cache entries are {} (not able to extract
    # ligand, for example).
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


def _build_moad_cache_file(
        filename: str,
        moad: "MOADInterface",
        cache_items_to_update: CacheItemsToUpdate,
        cores: int = None,
):
    # This builds/updates the whole BindingMOAD cache (on disk).

    # TODO: hard coding to use all cores
    cores = None

    # Load existing cache if it exists. So you can add to it.
    if os.path.exists(filename):
        with open(filename) as f:
            cache = json.load(f)
    else:
        # No existing cache, so start empty.
        cache = {}

    # Setup and modify cache items to update.
    global CACHE_ITEMS_TO_UPDATE
    CACHE_ITEMS_TO_UPDATE = cache_items_to_update

    _set_cache_params_to_update(cache)

    if not CACHE_ITEMS_TO_UPDATE.updatable():
        # Nothing to update
        return cache

    # global MOAD_REF
    # MOAD_REF = moad

    pdb_ids_queue = moad.targets
    list_ids_moad = []
    for pdb_id in moad.targets:
        list_ids_moad.append((pdb_id, moad[pdb_id], CACHE_ITEMS_TO_UPDATE))

    pbar = tqdm(total=len(pdb_ids_queue), desc="Building MOAD cache")
    with multiprocessing.Pool(cores) as p:
        for pdb_id, lig_infs in p.starmap(_get_info_given_pdb_id, list_ids_moad):
            pdb_id = pdb_id.lower()

            if pdb_id not in cache:
                cache[pdb_id] = {}

            for lig_name in lig_infs.keys():
                if lig_name not in cache[pdb_id]:
                    cache[pdb_id][lig_name] = {}
                lig_inf = lig_infs[lig_name]
                for prop_name in lig_inf.keys():
                    prop_val = lig_inf[prop_name]
                    cache[pdb_id][lig_name][prop_name] = prop_val

            pbar.update(1)

    pbar.close()

    # Save cache with updated information.
    with open(filename, "w") as f:
        json.dump(cache, f, indent=4)

    return cache


def load_cache_and_filter(
        lig_filter_func: Any,  # The function used to filter the ligands
        moad: "MOADInterface",
        split: "MOAD_split",
        args: argparse.Namespace,
        make_dataset_entries_func: Any,
        cache_items_to_update: CacheItemsToUpdate,
        cache_file: Optional[Union[str, Path]] = None,
        cores: int = 1,
):
    # This not only builds/gets the cache, but also filters the properties to
    # retain only those BindingMOAD entries for training. For example, could
    # filter by ligand mass.

    # This function gets called from the dataset (e.g., fragment_dataset.py),
    # where the actual filters are set (via lig_filter_func).

    # Returns tuple, cache and filtered_cache.

    cache_file = Path(cache_file) if cache_file is not None else None
    cache = _build_moad_cache_file(str(cache_file), moad, cache_items_to_update, cores=cores)

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

            # Enforce filters.
            fails_filter = False
            for lig in moad[pdb_id].ligands:
                if lig.name != lig_name:
                    continue

                # You've found the ligand.
                if lig.smiles not in split.smiles:
                    # It is not in the split, so always skip it.
                    fails_filter = True
                    break

                if not lig_filter_func(args, lig, lig_inf):
                    # You've found the ligand, but it doesn't pass the filter.
                    fails_filter = True
                    break

            if fails_filter:
                continue

            # Add to filtered cache.
            filtered_cache.extend(
                make_dataset_entries_func(args, pdb_id, lig_name, lig_inf)
            )

    if not filtered_cache:
        raise Exception(
            "No ligands passed the moad filters. Could be that filters are too strict, or perhaps there is a problem with your CSV file. Representative ligand info to help with debugging: "
            + str(lig)
        )

    return cache, filtered_cache
