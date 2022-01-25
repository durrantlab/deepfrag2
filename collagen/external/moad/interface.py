
from dataclasses import field
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import numpy as np

from .types import (
    MOAD_split,
    MOAD_family,
    MOAD_class,
    MOAD_ligand,
    MOAD_target
)

split_rand_num_gen = None

def _split_seq(seq, p):
    global split_rand_num_gen
    l = list(seq)
    sz = len(l)
    split_rand_num_gen.shuffle(l)
    # np.random.shuffle(l)

    return l[: int(sz * p)], l[int(sz * p) :]


def _flatten(seq):
    a = []
    for s in seq:
        a += s
    return a


def _div2(seq):
    global split_rand_num_gen
    l = list(seq)
    sz = len(l)

    split_rand_num_gen.shuffle(l)
    # np.random.shuffle(l)

    return (set(l[: sz // 2]), set(l[sz // 2 :]))


def _div3(seq):
    global split_rand_num_gen

    l = list(seq)
    sz = len(l)

    split_rand_num_gen.shuffle(l)
    # np.random.shuffle(l)

    v = sz // 3
    return (set(l[:v]), set(l[v : v * 2]), set(l[v * 2 :]))


class MOADInterface(object):
    """
    Base class for interacting with Binding MOAD data. Initialize by passing the path to
    "every.csv" and the path to a folder containing structure files (can be nested).

    Args:
        metadata: Path to the metadata "every.csv" file.
        structures: Path to a folder container structure files.
    """

    classes: List["MOAD_class"]
    _lookup: Dict["str", "MOAD_target"] = field(default_factory=dict)
    _all_targets: List["str"] = field(default_factory=list)

    def __init__(self, metadata: Union[str, Path], structures: Union[str, Path]):
        self.classes = MOADInterface._load_classes(metadata)
        self._lookup = {}
        self._all_targets = []

        self._init_lookup()
        self._resolve_paths(structures)

    def _init_lookup(self):
        for c in self.classes:
            for f in c.families:
                for t in f.targets:
                    self._lookup[t.pdb_id.lower()] = t

        self._all_targets = [k for k in self._lookup]

    @property
    def targets(self) -> List["str"]:
        return self._all_targets

    def __getitem__(self, key: str) -> "MOAD_target":
        """
        Fetch a specific target by PDB ID.

        Args:
            key (str): A PDB ID (case-insensitive).

        Returns:
            MOAD_target: a MOAD_target object if found.
        """
        assert type(key) is str, f"PDB ID must be a str (got {type(key)})"
        k = key.lower()
        assert k in self._lookup, f'Target "{k}" not found.'
        return self._lookup[k]

    @staticmethod
    def _load_classes(path):
        with open(path, "r") as f:
            dat = f.read().strip().split("\n")

        classes = []
        curr_class = None
        curr_family = None
        curr_target = None

        for line in dat:
            parts = line.split(",")

            if parts[0] != "":  # 1: Protein Class
                if curr_class is not None:
                    classes.append(curr_class)
                curr_class = MOAD_class(ec_num=parts[0], families=[])
            elif parts[1] != "":  # 2: Protein Family
                if curr_target is not None:
                    curr_family.targets.append(curr_target)
                if curr_family is not None:
                    curr_class.families.append(curr_family)
                curr_family = MOAD_family(rep_pdb_id=parts[2], targets=[])
                curr_target = MOAD_target(pdb_id=parts[2], ligands=[])
            elif parts[2] != "":  # 3: Protein target
                if curr_target is not None:
                    curr_family.targets.append(curr_target)
                curr_target = MOAD_target(pdb_id=parts[2], ligands=[])
            elif parts[3] != "":  # 4: Ligand
                curr_target.ligands.append(
                    MOAD_ligand(
                        name=parts[3],
                        validity=parts[4],
                        affinity_measure=parts[5],
                        affinity_value=parts[7],
                        affinity_unit=parts[8],
                        smiles=parts[9],
                    )
                )

        if curr_target is not None:
            curr_family.targets.append(curr_target)
        if curr_family is not None:
            curr_class.families.append(curr_family)
        if curr_class is not None:
            classes.append(curr_class)

        return classes

    def _resolve_paths(self, path: Union[str, Path]):
        path = Path(path)

        files = {}
        for f in path.glob("./**/*.bio*"):
            pdbid = f.stem
            if not pdbid in files:
                files[pdbid] = []
            files[pdbid].append(f)

        for c in self.classes:
            for f in c.families:
                for t in f.targets:
                    k = t.pdb_id.lower()
                    if k in files:
                        t.files = sorted(files[k])
                    else:
                        # No structures for this pdb id!
                        pass

    def _smiles_for(self, targets: List[str]) -> Set[str]:
        """Return all the SMILES strings contained in the selected targets."""
        smiles = set()

        for target in targets:
            for ligand in self[target].ligands:
                smi = ligand.smiles
                if ligand.is_valid and smi not in ["n/a", "NULL"]:
                    smiles.add(smi)

        return smiles

    def compute_split(
        self,
        seed: int = 0,
        fraction_train: float = 0.6,
        fraction_val: float = 0.5,
        prevent_smiles_overlap: bool = True,
        save_splits: str = None,
        load_splits: str = None
    ) -> Tuple["MOAD_split", "MOAD_split", "MOAD_split"]:
        """Compute a TRAIN/VAL/TEST split.

        Targets are first assigned to a TRAIN set with `p_train` probability. The remaining targets are assigned to a VAL set with
        `p_val` probability. The unused targets are assigned to the TEST set.

        Args:
            seed (int, optional): If set to a nonzero number, compute_split will always return the same split.
            p_train (float, optional): Percentage of targets to use in the TRAIN set.
            p_val (float, optional): Percentage of (non-train) targets to use in the VAL set.

        Returns:
            Tuple[MOAD_split, MOAD_split, MOAD_split]: train/val/test sets
        """

        if seed != 0:
            global split_rand_num_gen
            split_rand_num_gen = np.random.default_rng(seed)

            # Note: Below also makes rotations and other randomly determined
            # aspects of the code deterministic. So using
            # np.random.default_rng(seed) instead.
            # np.random.seed(seed)

        train_pdb_ids = None
        val_pdb_ids = None
        test_pdb_ids = None
        train_smi = None
        val_smi = None
        test_smi = None

        if load_splits is None:
            # Not loading splits, so generate based on random seed.
            families: List[List[str]] = []
            # import pdb; pdb.set_trace()
            for c in self.classes:  # [:250]:
                for f in c.families:
                    families.append([x.pdb_id for x in f.targets])

            # These are lists of lists, where each inner list contains all the
            # members of the corresponding family. Grouped this way so memboers of
            # same family are spread across train, val, test sets.
            train_families, other_families = _split_seq(families, fraction_train)
            val_families, test_families = _split_seq(other_families, fraction_val)

            train_pdb_ids = _flatten(train_families)
            val_pdb_ids = _flatten(val_families)
            test_pdb_ids = _flatten(test_families)

            train_smi = self._smiles_for(train_pdb_ids)
            val_smi = self._smiles_for(val_pdb_ids)
            test_smi = self._smiles_for(test_pdb_ids)

            if prevent_smiles_overlap:
                # Reassign overlapping SMILES.
                a_train, a_val = _div2((train_smi & val_smi) - test_smi)
                b_val, b_test = _div2((val_smi & test_smi) - train_smi)
                c_train, c_test = _div2((train_smi & test_smi) - val_smi)
                d_train, d_val, d_test = _div3((train_smi & val_smi & test_smi))

                train_smi = (train_smi - (val_smi | test_smi)) | a_train | c_train | d_train
                val_smi = (val_smi - (train_smi | test_smi)) | a_val | b_val | d_val
                test_smi = (test_smi - (train_smi | val_smi)) | b_test | c_test | d_test

        else:
            # Loading splits. Get from the file.
            split_inf = json.load(open(load_splits))
            train_pdb_ids = split_inf["train"]["pdbs"]
            val_pdb_ids = split_inf["val"]["pdbs"]
            test_pdb_ids = split_inf["test"]["pdbs"]
            train_smi = set(split_inf["train"]["smiles"])
            val_smi = set(split_inf["val"]["smiles"])
            test_smi = set(split_inf["test"]["smiles"])

            # Reset seed just in case you also use save_splits. Not used.
            seed = split_inf["test"]  

        if save_splits is not None:
            # Save spits and seed to json.
            split_inf = {
                "seed": seed,
                "train": {
                    "pdbs": train_pdb_ids,
                    "smiles": [smi for smi in train_smi]
                    },
                "val": {
                    "pdbs": val_pdb_ids,
                    "smiles": [smi for smi in val_smi]
                    },
                "test": {
                    "pdbs": test_pdb_ids,
                    "smiles": [smi for smi in test_smi]
                    },
            }
            json.dump(split_inf, open(save_splits, "w"), indent=4)

        return (
            MOAD_split(name="TRAIN", targets=train_pdb_ids, smiles=train_smi),
            MOAD_split(name="VAL", targets=val_pdb_ids, smiles=val_smi),
            MOAD_split(name="TEST", targets=test_pdb_ids, smiles=test_smi),
        )

    def full_split(self) -> MOAD_split:
        """Returns a split containing all targets and smiles strings."""
        return MOAD_split(
            name="Full",
            targets=self.targets,
            smiles=list(self._smiles_for(self.targets)),
        )
