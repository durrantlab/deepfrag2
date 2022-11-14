from dataclasses import dataclass
from typing import List, Set, Tuple
from .types import MOAD_split
import numpy as np
from collagen.util import sorted_list
import json
from collagen.external.moad.split_clustering import generate_splits_from_clustering

# This module has functions required to split the MOAD into train, val, and test
# sets.

split_rand_num_gen = None


@dataclass
class MOAD_splits_pdb_ids:
    train: List
    val: List
    test: List


@dataclass
class MOAD_splits_smiles:
    train: Set[str]
    val: Set[str]
    test: Set[str]


def _split_seq_per_probability(seq, p):
    # Divide a sequence according to a probability, p.

    global split_rand_num_gen
    l = sorted_list(seq)
    size = len(l)
    split_rand_num_gen.shuffle(l)
    # np.random.shuffle(l)

    return l[: int(size * p)], l[int(size * p) :]


def _flatten(seq):
    a = []
    for s in seq:
        a += s
    return a


def _divide_into_two_parts(seq):
    global split_rand_num_gen
    l = sorted_list(seq)  # To make deterministic is same seed used
    size = len(l)
    half_size = size // 2  # same as int(size / 2.0)

    split_rand_num_gen.shuffle(l)

    # np.random.shuffle(l)

    return (set(l[:half_size]), set(l[half_size:]))


def _divide_into_three_parts(seq):
    global split_rand_num_gen
    l = sorted_list(seq)
    size = len(l)

    split_rand_num_gen.shuffle(l)
    # np.random.shuffle(l)

    thid_size = size // 3
    return (
        set(l[:thid_size]),
        set(l[thid_size : thid_size * 2]),
        set(l[thid_size * 2 :]),
    )


def _smiles_for(moad: "MOADInterface", targets: List[str]) -> Set[str]:
    """Return all the SMILES strings contained in the selected targets."""

    smiles = set()

    for target in targets:
        for ligand in moad[target].ligands:
            smi = ligand.smiles
            if ligand.is_valid and smi not in ["n/a", "NULL"]:
                smiles.add(smi)

    return smiles


def _limit_split_size(
    max_pdbs_train: int,
    max_pdbs_val: int,
    max_pdbs_test: int,
    pdb_ids: MOAD_splits_pdb_ids,
) -> MOAD_splits_pdb_ids:
    # If the user has asked to limit the size of the train, test, or val set,
    # impose those limits here.

    if max_pdbs_train is not None and len(pdb_ids.train) > max_pdbs_train:
        pdb_ids.train = pdb_ids.train[:max_pdbs_train]

    if max_pdbs_val is not None and len(pdb_ids.val) > max_pdbs_val:
        pdb_ids.val = pdb_ids.val[:max_pdbs_val]

    if max_pdbs_test is not None and len(pdb_ids.test) > max_pdbs_test:
        pdb_ids.test = pdb_ids.test[:max_pdbs_test]

    return pdb_ids


def _generate_splits_from_scratch(
    moad: "MOADInterface",
    fraction_train: float = 0.6,
    fraction_val: float = 0.5,
    prevent_smiles_overlap: bool = True,
    max_pdbs_train: int = None,
    max_pdbs_val: int = None,
    max_pdbs_test: int = None,
    butina_cluster_division: bool = False,
    butina_cluster_cutoff: float = 0.4,
):
    if not butina_cluster_division:
        print("Building training/validation/test sets in a random way")
        # Not loading previously determined splits from disk, so generate based on
        # random seed.

        # First, get a flat list of all the families (not grouped by class).
        families: List[List[str]] = []
        for c in moad.classes:
            for f in c.families:
                families.append([x.pdb_id for x in f.targets])

        # Note that we're working with families (not individual targets in those
        # families) so members of same family are shared across train, val, test
        # sets.

        # Divide the families into train/val/test sets.
        train_families, other_families = _split_seq_per_probability(
            families, fraction_train
        )
        val_families, test_families = _split_seq_per_probability(
            other_families, fraction_val
        )
    else:
        print("Building training/validation/test sets based on Butina clustering")
        train_families, val_families, test_families = generate_splits_from_clustering(moad, split_rand_num_gen, fraction_train, fraction_val, butina_cluster_cutoff,)

    # Now that they are divided, we can keep only the targets themselves (no
    # longer organized into families).
    pdb_ids = MOAD_splits_pdb_ids(
        train=_flatten(train_families),
        val=_flatten(val_families),
        test=_flatten(test_families),
    )

    # If the user has asked to limit the size of the train, test, or val set,
    # impose those limits here.
    pdb_ids = _limit_split_size(
        max_pdbs_train,
        max_pdbs_val,
        max_pdbs_test,
        pdb_ids,
    )

    # Get all the smiles associated with the targets in each set.
    all_smis = MOAD_splits_smiles(
        train=_smiles_for(moad, pdb_ids.train),
        val=_smiles_for(moad, pdb_ids.val),
        test=_smiles_for(moad, pdb_ids.test),
    )

    if prevent_smiles_overlap:
        # Reassign overlapping SMILES.
        train_val = all_smis.train & all_smis.val
        val_test = all_smis.val & all_smis.test
        train_test = all_smis.train & all_smis.test
        train_val_not_test = train_val - all_smis.test
        val_test_not_train = val_test - all_smis.train
        train_test_not_val = train_test - all_smis.val
        train_test_val = all_smis.train & all_smis.val & all_smis.test

        a_train, a_val = _divide_into_two_parts(train_val_not_test)
        b_val, b_test = _divide_into_two_parts(val_test_not_train)
        c_train, c_test = _divide_into_two_parts(train_test_not_val)
        d_train, d_val, d_test = _divide_into_three_parts(train_test_val)

        all_smis.train = (
            (all_smis.train - (all_smis.val | all_smis.test))
            | a_train
            | c_train
            | d_train
        )
        all_smis.val = (
            (all_smis.val - (all_smis.train | all_smis.test)) | a_val | b_val | d_val
        )
        all_smis.test = (
            (all_smis.test - (all_smis.train | all_smis.val)) | b_test | c_test | d_test
        )

    return pdb_ids, all_smis


def _load_splits_from_disk(
    moad: "MOADInterface",
    load_splits: str = None,
    max_pdbs_train: int = None,
    max_pdbs_val: int = None,
    max_pdbs_test: int = None,
):
    # User has asked to load splits from file on disk. Get from the file.
    with open(load_splits) as f:
        split_inf = json.load(f)

    pdb_ids = MOAD_splits_pdb_ids(
        train=split_inf["train"]["pdbs"],
        val=split_inf["val"]["pdbs"],
        test=split_inf["test"]["pdbs"],
    )

    # Reset seed just in case you also use save_splits. Not used.
    seed = split_inf["test"]

    if max_pdbs_train is None and max_pdbs_val is None and max_pdbs_test is None:
        # Load from cache
        all_smis = MOAD_splits_smiles(
            train=set(split_inf["train"]["smiles"]),
            val=set(split_inf["val"]["smiles"]),
            test=set(split_inf["test"]["smiles"]),
        )

        return pdb_ids, all_smis, seed

    # If you get here, the user has asked to limit the number of pdbs in the
    # train/test/val set(s), so also don't get the smiles from the cache as
    # above.
    pdb_ids = _limit_split_size(
        max_pdbs_train,
        max_pdbs_val,
        max_pdbs_test,
        pdb_ids,
    )

    all_smis = MOAD_splits_smiles(
        train=_smiles_for(moad, pdb_ids.train),
        val=_smiles_for(moad, pdb_ids.val),
        test=_smiles_for(moad, pdb_ids.test),
    )

    return pdb_ids, all_smis, seed


def _save_split(
    save_splits: str,
    seed: int,
    pdb_ids: MOAD_splits_pdb_ids,
    all_smis: MOAD_splits_smiles,
):
    # Save spits and seed to json (for record keeping).
    split_inf = {
        "seed": seed,
        "unique_counts": {
            "train": {
                "pdbs": len(set(pdb_ids.train)),
                "frags": len(set(all_smis.train)),
            },
            "val": {
                "pdbs": len(str(pdb_ids.val)),
                "frags": len(set(all_smis.val)),
            },
            "test": {
                "pdbs": len(set(pdb_ids.test)),
                "frags": len(set(all_smis.test)),
            },
        },
        "train": {
            "pdbs": pdb_ids.train,
            "smiles": [smi for smi in all_smis.train],
        },
        "val": {"pdbs": pdb_ids.val, "smiles": [smi for smi in all_smis.val]},
        "test": {"pdbs": pdb_ids.test, "smiles": [smi for smi in all_smis.test]},
    }
    with open(save_splits, "w") as f:
        json.dump(split_inf, f, indent=4)


def compute_moad_split(
    moad: "MOADInterface",
    seed: int = 0,
    fraction_train: float = 0.6,
    fraction_val: float = 0.5,
    prevent_smiles_overlap: bool = True,
    save_splits: str = None,
    load_splits: str = None,
    max_pdbs_train: int = None,
    max_pdbs_val: int = None,
    max_pdbs_test: int = None,
    butina_cluster_division: bool = False,
    butina_cluster_cutoff = 0.4,
) -> Tuple["MOAD_split", "MOAD_split", "MOAD_split"]:
    """Compute a TRAIN/VAL/TEST split.

    Targets are first assigned to a TRAIN set with `p_train` probability.
    The remaining targets are assigned to a VAL set with `p_val`
    probability. The unused targets are assigned to the TEST set.

    Args:
        seed (int, optional): If set to a nonzero number, compute_MOAD_split
            will always return the same split.
        p_train (float, optional): Percentage of targets to use in the
            TRAIN set.
        p_val (float, optional): Percentage of (non-train) targets to use
            in the VAL set.

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

    # pdb_ids = MOAD_splits_pdb_ids(None, None, None)
    # all_smis = MOAD_splits_smiles(None, None, None)

    # train_pdb_ids = None
    # val_pdb_ids = None
    # test_pdb_ids = None
    # train_all_smis = None
    # val_all_smis = None
    # test_all_smis = None

    if load_splits is None:
        # Not loading previously determined splits from disk, so generate
        # based on random seed.
        pdb_ids, all_smis = _generate_splits_from_scratch(
            moad,
            fraction_train,
            fraction_val,
            prevent_smiles_overlap,
            max_pdbs_train,
            max_pdbs_val,
            max_pdbs_test,
            butina_cluster_division,
            butina_cluster_cutoff,
        )
    else:
        # User has asked to load splits from file on disk. Get from the file.
        pdb_ids, all_smis, seed = _load_splits_from_disk(
            moad,
            load_splits,
            max_pdbs_train,
            max_pdbs_val,
            max_pdbs_test,
        )

    if save_splits is not None:
        # Save spits and seed to json (for record keeping).
        _save_split(save_splits, seed, pdb_ids, all_smis)

    return (
        MOAD_split(name="TRAIN", targets=pdb_ids.train, smiles=all_smis.train),
        MOAD_split(name="VAL", targets=pdb_ids.val, smiles=all_smis.val),
        MOAD_split(name="TEST", targets=pdb_ids.test, smiles=all_smis.test),
    )


def full_moad_split(moad: "MOADInterface") -> MOAD_split:
    """Returns a split containing all targets and smiles strings."""

    return MOAD_split(
        name="Full",
        targets=moad.targets,
        smiles=sorted_list(moad._smiles_for(moad.targets)),
    )
