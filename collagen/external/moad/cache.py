from __future__ import annotations
from typing import TYPE_CHECKING

from collagen.external.moad.moad_utils import fix_moad_smiles

if TYPE_CHECKING:
    from collagen.external.moad.moad_interface import MOADInterface

import multiprocessing
import os
import json
from typing import Any
from tqdm.std import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
from scipy.spatial.distance import cdist
import numpy as np


class CacheItemsToUpdate(object):
    # Updatable
    lig_mass: bool
    frag_masses: bool
    murcko_scaffold: bool
    num_heavy_atoms: bool  # TODO: Test this
    frag_dist_to_recep: bool  # TODO: Test this

    # Updatable
    def __init__(
        self,
        lig_mass=False,
        frag_masses=False,
        murcko_scaffold=False,
        num_heavy_atoms=False,
        frag_dist_to_recep=False,
    ):
        self.lig_mass = lig_mass
        self.frag_masses = frag_masses
        self.murcko_scaffold = murcko_scaffold
        self.num_heavy_atoms = num_heavy_atoms
        self.frag_dist_to_recep = frag_dist_to_recep

    def updatable(self) -> bool:
        # Updatable
        return True in [
            self.lig_mass,
            self.frag_masses,
            self.murcko_scaffold,
            self.num_heavy_atoms,
            self.frag_dist_to_recep,
        ]


MOAD_REF: MOADInterface
CACHE_ITEMS_TO_UPDATE = CacheItemsToUpdate()


def get_info_given_pdb_id(pdb_id: str):
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

                # Updatable
                if CACHE_ITEMS_TO_UPDATE.lig_mass:
                    try:
                        lig_infs[lig_name]["lig_mass"] = int(lig.mass)
                    except:
                        lig_infs[lig_name]["lig_mass"] = 999999

                if CACHE_ITEMS_TO_UPDATE.frag_masses:
                    try:
                        lig_infs[lig_name]["frag_masses"] = [
                            int(x[1].mass) for x in lig.split_bonds()
                        ]
                    except:
                        lig_infs[lig_name]["frag_masses"] = []

                if CACHE_ITEMS_TO_UPDATE.murcko_scaffold:
                    try:
                        smi_fixed = fix_moad_smiles(lig.smiles(True))
                        scaffold_smi = MurckoScaffoldSmilesFromSmiles(
                            smi_fixed, includeChirality=True
                        )
                        lig_infs[lig_name]["murcko_scaffold"] = scaffold_smi
                        # str(dir(lig)) # scaffold_smi
                    except:
                        lig_infs[lig_name]["murcko_scaffold"] = ""

                if CACHE_ITEMS_TO_UPDATE.num_heavy_atoms:
                    try:
                        lig_infs[lig_name]["num_heavy_atoms"] = lig.num_heavy_atoms
                    except:
                        lig_infs[lig_name]["num_heavy_atoms"] = 999999

                if CACHE_ITEMS_TO_UPDATE.frag_dist_to_recep:
                    try:
                        min_dist = np.min(cdist(lig.coords, receptor.coords))
                        lig_infs[lig_name]["frag_dist_to_recep"] = min_dist
                    except:
                        lig_infs[lig_name]["frag_dist_to_recep"] = 999999

        except:
            pass

    return (pdb_id, lig_infs)


def build_moad_cache(
    filename: str,
    moad: MOADInterface,
    cache_items_to_update: CacheItemsToUpdate,
    cores: int = None,
):
    # TODO: hard coding to use all cores
    cores = None

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
    CACHE_ITEMS_TO_UPDATE = cache_items_to_update

    if len(cache.keys()) > 0:
        first_pdb = cache[list(cache.keys())[0]]
        first_lig = first_pdb[list(first_pdb.keys())[0]]

        # Updatable
        if "lig_mass" in first_lig:
            CACHE_ITEMS_TO_UPDATE.lig_mass = False
        if "frag_masses" in first_lig:
            CACHE_ITEMS_TO_UPDATE.frag_masses = False
        if "murcko_scaffold" in first_lig:
            CACHE_ITEMS_TO_UPDATE.murcko_scaffold = False
        if "num_heavy_atoms" in first_lig:
            CACHE_ITEMS_TO_UPDATE.num_heavy_atoms = False
        if "frag_dist_to_recep" in first_lig:
            CACHE_ITEMS_TO_UPDATE.frag_dist_to_recep = False

    if CACHE_ITEMS_TO_UPDATE.updatable():
        pdb_ids_queue = MOAD_REF.targets
        pbar = tqdm(total=len(pdb_ids_queue), desc="Building MOAD cache")
        with multiprocessing.Pool(cores) as p:
            for pdb_id, lig_infs in p.imap_unordered(
                get_info_given_pdb_id, pdb_ids_queue
            ):
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
