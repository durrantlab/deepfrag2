from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collagen.external.moad.moad_interface import MOADInterface

import multiprocessing
import os
import json
from typing import Any
from tqdm.std import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles


class CacheItemsToUpdate(object):
    # Updatable
    lig_mass: bool
    frag_masses: bool
    # has_murcko_scaffold: bool

    def updatable(self) -> bool:
        # Updatable
        return True in [self.lig_mass, self.frag_masses]


MOAD_REF: MOADInterface
CACHE_ITEMS_TO_UPDATE = CacheItemsToUpdate()


def get_info_given_pdb(pdb_id: str):
    global MOAD_REF
    global CACHE_ITEMS_TO_UPDATE

    moad_entry_info = MOAD_REF[pdb_id]

    lig_infs = {}
    for lig_chunk_idx in range(len(moad_entry_info)):
        try:

            # Unpack info to get ligands
            receptor, ligands = moad_entry_info[lig_chunk_idx]

            for lig in ligands:
                lig_name = lig.meta["moad_ligand"].name
                lig_infs[lig_name] = {"lig_chunk_idx": lig_chunk_idx}

                if CACHE_ITEMS_TO_UPDATE.lig_mass:
                    lig_infs[lig_name]["lig_mass"] = int(lig.mass)
                if CACHE_ITEMS_TO_UPDATE.frag_masses:
                    try:
                        lig_infs[lig_name]["frag_masses"] = [
                            int(x[1].mass) for x in lig.split_bonds()
                        ]
                    except:
                        lig_infs[lig_name]["frag_masses"] = []

                # lig_to_mass[lig.meta["moad_ligand"].name] = int(lig.mass)
                # lig_infs.append(lig_name)
        except:
            pass

    return (pdb_id, lig_infs)


def build_moad_cache(
    filename: str,
    moad: MOADInterface,
    lig_mass: bool = False,
    frag_masses: bool = False,
    cores: int = None,
):

    global MOAD_REF
    MOAD_REF = moad

    # Load existing cache if it exists.
    if os.path.exists(filename):
        with open(filename) as f:
            cache = json.load(f)
    else:
        cache = {}

    # Setup and modify cache items to update.
    global CACHE_ITEMS_TO_UPDATE

    # Updatable
    CACHE_ITEMS_TO_UPDATE.lig_mass = lig_mass
    CACHE_ITEMS_TO_UPDATE.frag_masses = frag_masses

    if len(cache.keys()) > 0:
        first_pdb = cache[list(cache.keys())[0]]
        first_lig = first_pdb[list(first_pdb.keys())[0]]

        # Updatable
        if "lig_mass" in first_lig:
            CACHE_ITEMS_TO_UPDATE.lig_mass = False
        if "frag_masses" in first_lig:
            CACHE_ITEMS_TO_UPDATE.frag_masses = False

    if CACHE_ITEMS_TO_UPDATE.updatable():
        pdb_ids_queue = MOAD_REF.targets
        pbar = tqdm(total=len(pdb_ids_queue), desc="Building MOAD cache")
        with multiprocessing.Pool(cores) as p:
            for pdb_id, lig_infs in p.imap_unordered(get_info_given_pdb, pdb_ids_queue):
                pdb_id = pdb_id.lower()

                if not pdb_id in cache:
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

        with open(filename, "w") as f:
            json.dump(cache, f, indent=4)

    return cache
