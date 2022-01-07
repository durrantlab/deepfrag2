
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


def _split_seq(seq, p):
    l = list(seq)
    sz = len(l)
    np.random.shuffle(l)

    return l[: int(sz * p)], l[int(sz * p) :]


def _flatten(seq):
    a = []
    for s in seq:
        a += s
    return a


def _div2(seq):
    l = list(seq)
    sz = len(l)

    np.random.shuffle(l)

    return (set(l[: sz // 2]), set(l[sz // 2 :]))


def _div3(seq):
    l = list(seq)
    sz = len(l)

    np.random.shuffle(l)

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
        p_train: float = 0.6,
        p_val: float = 0.5,
        prevent_smiles_overlap: bool = True,
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
            np.random.seed(seed)

        families: List[List[str]] = []
        # import pdb; pdb.set_trace()
        for c in self.classes:  # [:250]:
            for f in c.families:
                families.append([x.pdb_id for x in f.targets])

        train_f, other_f = _split_seq(families, p_train)
        val_f, test_f = _split_seq(other_f, p_val)

        train_ids = _flatten(train_f)
        val_ids = _flatten(val_f)
        test_ids = _flatten(test_f)

        train_smi = self._smiles_for(train_ids)
        val_smi = self._smiles_for(val_ids)
        test_smi = self._smiles_for(test_ids)

        if prevent_smiles_overlap:
            # Reassign overlapping SMILES.
            a_train, a_val = _div2((train_smi & val_smi) - test_smi)
            b_val, b_test = _div2((val_smi & test_smi) - train_smi)
            c_train, c_test = _div2((train_smi & test_smi) - val_smi)
            d_train, d_val, d_test = _div3((train_smi & val_smi & test_smi))

            train_smi = (train_smi - (val_smi | test_smi)) | a_train | c_train | d_train
            val_smi = (val_smi - (train_smi | test_smi)) | a_val | b_val | d_val
            test_smi = (test_smi - (train_smi | val_smi)) | b_test | c_test | d_test

        # Save spit and seed to json in working directory if running in docker
        # container.
        if os.path.exists("/working/"):
            split_inf = {
                "seed": seed,
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            }
            json.dump(split_inf, open("/working/moad_split.json", "w"), indent=4)

        return (
            MOAD_split(name="TRAIN", targets=train_ids, smiles=train_smi),
            MOAD_split(name="VAL", targets=val_ids, smiles=val_smi),
            MOAD_split(name="TEST", targets=test_ids, smiles=test_smi),
        )

    def full_split(self) -> MOAD_split:
        """Returns a split containing all targets and smiles strings."""
        return MOAD_split(
            name="Full",
            targets=self.targets,
            smiles=list(self._smiles_for(self.targets)),
        )
