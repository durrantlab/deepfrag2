"""Functions required to split the MOAD into train, val, and test sets."""

import os
from dataclasses import dataclass
from typing import List, Set, Tuple
from .types import MOAD_split
import numpy as np
from collagen.util import sorted_list
import json
from collagen.external.moad.split_clustering import generate_splits_from_clustering
from tqdm import tqdm


split_rand_num_gen = None


@dataclass
class MOAD_splits_pdb_ids:

    """MOAD splits in terms of PDB IDs."""

    train: List
    val: List
    test: List


@dataclass
class MOAD_splits_smiles:

    """MOAD splits in terms of SMILES."""

    train: Set[str]
    val: Set[str]
    test: Set[str]


def _split_seq_per_probability(seq: List, p: float) -> Tuple[List, List]:
    """Divide a sequence according to a probability, p.

    Args:
        seq (List): Sequence to be divided.
        p (float): Probability of the first part.

    Returns:
        Tuple[List, List]: First and second parts of the sequence.
    """
    global split_rand_num_gen
    l = sorted_list(seq)
    size = len(l)
    split_rand_num_gen.shuffle(l)
    # np.random.shuffle(l)

    return l[: int(size * p)], l[int(size * p) :]


def _flatten(seq: List[List]) -> List:
    """Flatten a list of lists.

    Args:
        seq (List[List]): List of lists.

    Returns:
        List: Flattened list.
    """
    a = []
    for s in seq:
        a += s
    return a


def _random_divide_two_prts(seq: List) -> Tuple[List, List]:
    """Divide a sequence into two parts.

    Args:
        seq (List): Sequence to be divided.

    Returns:
        Tuple[List, List]: First and second parts of the sequence.
    """
    global split_rand_num_gen
    l = sorted_list(seq)  # To make deterministic is same seed used
    size = len(l)
    half_size = size // 2  # same as int(size / 2.0)

    split_rand_num_gen.shuffle(l)

    # np.random.shuffle(l)

    return (set(l[:half_size]), set(l[half_size:]))


def _random_divide_three_prts(seq: List) -> Tuple[List, List, List]:
    """Divide a sequence into three parts.

    Args:
        seq (List): Sequence to be divided.

    Returns:
        Tuple[List, List, List]: First, second, and third parts of the sequence.
    """
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
    """Return all the SMILES strings contained in the selected targets.
    
    Args:
        moad (MOADInterface): MOADInterface object.
        targets (List[str]): List of targets.
        
    Returns:
        Set[str]: Set of SMILES strings.
    """
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

def jdd_approach(moad: "MOADInterface"):
    # First, get a flat list of all the families (not grouped by class).
    families: List[List[str]] = []
    for c in moad.classes:
        families.extend([x.pdb_id for x in f.targets] for f in c.families)

    #### JDD EXPERIMENTING

    # For each of the familes, get the smiles strings for all the ligands
    smiles: List[List[str]] = [_smiles_for(moad, family) for family in families]

    # Now merge and flatten these lists into a list of lists, where the
    # inner list is [pdb_id, family_idx, smiles]
    complex_infos = []
    for family_idx, family in enumerate(families):
        complex_infos.extend(
            [pdb_id, family_idx, smi]
            for pdb_id, smi in zip(family, smiles[family_idx])
        )

    # Unique PDB IDS (about 33,000). But 41,000 in directory and every.csv.
    # Why the difference? Which ones are missing? Some ligands may have no
    # fragments. Could that be it? I think you need to print out the lists
    # and compare.

    # import pdb; pdb.set_trace()

    def move_to_current_cluster(item):
        pdb_id, family_idx, smi = item
        current_cluster["family_idxs"].add(family_idx)
        current_cluster["smiles"].add(smi)
        current_cluster["items"].append(item)

    clusters = []
    while complex_infos:
        current_cluster = {
            "family_idxs": set([]),
            "smiles": set([]),
            "items": [],
        }

        # Get started by adding the first one
        move_to_current_cluster(complex_infos.pop())

        while True:
            print("")
            print("Number of clusters:", len(clusters))
            print("Number of complexes left to assign:", len(complex_infos))
            any_complex_assigned = False
            for complex_infos_idx, item in enumerate(complex_infos):
                if item is None:
                    continue
                pdb_id, family_idx, smi = item
                if family_idx in current_cluster["family_idxs"]:
                    # This family already in current cluster, so must add this item
                    # to same cluster.
                    move_to_current_cluster(item)
                    complex_infos[complex_infos_idx] = None
                    print(f"Added {pdb_id} to current cluster (same family)")
                    any_complex_assigned = True
                elif smi in current_cluster["smiles"]:
                    # This ligand already in current cluster, so must add this item
                    # to same cluster.
                    move_to_current_cluster(item)
                    complex_infos[complex_infos_idx] = None
                    print(f"Added {pdb_id} to current cluster (same ligand)")
                    any_complex_assigned = True
            complex_infos = [x for x in complex_infos if x is not None]
            if not any_complex_assigned:
                # No additional complexes assigned to current cluster, so must
                # be done.
                clusters.append(current_cluster)
                break
    
    counts = [len(cluster["items"]) for cluster in clusters]

    # Which cluster has the largest number of complexes?
    idx_of_biggest = np.argmax(counts)
    train_set = clusters[idx_of_biggest]

    # Merge all the remaining into a single set
    test_set = set()
    for i, cluster in enumerate(clusters):
        if i != idx_of_biggest:
            import pdb; pdb.set_trace()
            test_set.update(cluster["items"])

    # Validation and testing set same in this case
    val_set = test_set

    report_sizes(train_set, test_set, val_set)

    import pdb; pdb.set_trace()
    
    print("All together:", np.sum(counts))

def get_families_and_smiles(moad: "MOADInterface"):
    families: List[List[str]] = []
    for c in moad.classes:
        families.extend([x.pdb_id for x in f.targets] for f in c.families)

    smiles: List[List[str]] = [_smiles_for(moad, family) for family in families]

    return families, smiles

def report_sizes(train_set, test_set, val_set):
    print(f"Training set size: {len(train_set)}")
    print(f"Testing set size: {len(test_set)}")
    print(f"Validation set size: {len(val_set)}")

    # Get the smiles in each of the sets
    train_smiles = {complex['smiles'] for complex in train_set}
    test_smiles = {complex['smiles'] for complex in test_set}
    val_smiles = {complex['smiles'] for complex in val_set}

    # Get the families in each of the sets
    train_families = {complex['family_idx'] for complex in train_set}
    test_families = {complex['family_idx'] for complex in test_set}
    val_families = {complex['family_idx'] for complex in val_set}

    # Verify that there is no overlap between the sets
    print(f"Train and test overlap, SMILES: {len(train_smiles & test_smiles)}")
    print(f"Train and val overlap: {len(train_smiles & val_smiles)}")
    print(f"Test and val overlap, SMILES: {len(test_smiles & val_smiles)}")
    print(f"Train and test overlap, families: {len(train_families & test_families)}")
    print(f"Train and val overlap, families: {len(train_families & val_families)}")
    print(f"Test and val overlap, families: {len(test_families & val_families)}")

    # What is the number that were not assigned to any cluster?
    # print(f"Number of complexes not assigned to any cluster: {len(data) - len(train_set) - len(test_set) - len(val_set)}")


def _generate_splits_from_scratch(
    moad: "MOADInterface",
    fraction_train: float = 0.6,
    fraction_val: float = 0.5,
    prevent_smiles_overlap: bool = True,
    max_pdbs_train: int = None,
    max_pdbs_val: int = None,
    max_pdbs_test: int = None,
    butina_cluster_cutoff: float = 0.4,
):
    if butina_cluster_cutoff:
        print("Building training/validation/test sets based on Butina clustering")
        train_families, val_families, test_families = generate_splits_from_clustering(
            moad,
            split_rand_num_gen,
            fraction_train,
            fraction_val,
            butina_cluster_cutoff,
        )
    else:
        print("Building training/validation/test sets via random selection")
        # Not loading previously determined splits from disk, so generate based
        # on random seed.

        jdd_approach(moad)
        # chat_gpt4_approach(moad)
        
        import pdb; pdb.set_trace()




        #### END JDD EXPERIMENTING

        # Note that we're working with families (not individual targets in those
        # families) so members of same family are not shared across train, val,
        # test sets.

        # Divide the families into train/val/test sets.
        train_families, other_families = _split_seq_per_probability(
            families, fraction_train
        )
        val_families, test_families = _split_seq_per_probability(
            other_families, fraction_val
        )

    # Now that they are divided, we can keep only the targets themselves (no
    # longer organized into families).
    pdb_ids = MOAD_splits_pdb_ids(
        train=_flatten(train_families),
        val=_flatten(val_families),
        test=_flatten(test_families),
    )

    # If the user has asked to limit the size of the train, test, or val set,
    # impose those limits here.
    pdb_ids = _limit_split_size(max_pdbs_train, max_pdbs_val, max_pdbs_test, pdb_ids,)

    # Get all the ligand SMILES associated with the targets in each set.
    all_smis = MOAD_splits_smiles(
        train=_smiles_for(moad, pdb_ids.train),
        val=_smiles_for(moad, pdb_ids.val),
        test=_smiles_for(moad, pdb_ids.test),
    )

    if prevent_smiles_overlap:
        reassign_overlapping_smiles(all_smis)

    # TODO: Consider this GPT4 suggestion:

    # The problem with this approach is that even if smiles are independent, the
    # corresponding pdb ids are not also moved into the appropriate
    # train/test/val sets, so this data is thrown out elsewhere in the code. I
    # need some code that moves pdbs and the associated smiles together. And
    # yet, at the same time, it is still important that pdbs of the same family
    # are not split across the three sets, and it is still important that
    # identical smiles do not appear in the train/test/val sets. What new
    # approach do you recommend?

    # To address this issue, you can modify the approach to first group the data
    # by both protein family and ligand identity (SMILES) and then split the
    # groups into training, validation, and testing sets. This will ensure that
    # PDB IDs and their associated SMILES are moved together while maintaining
    # the required constraints. Here's a recommended approach:

    # 1. Group the data by protein family and ligand identity (SMILES):

    #   a. Create a dictionary with keys as tuples of protein family and SMILES,
    #   and values as lists of corresponding PDB IDs.

    # 2. Split the groups into training, validation, and testing sets:

    #   a. Use the same splitting function (e.g., `_split_seq_per_probability`)
    #   or any other method to split the dictionary keys (protein family, SMILES
    #   tuples) into training, validation, and testing groups.

    # 3. Flatten the PDB ID lists for each set:

    #   a. For each set (training, validation, testing), go through the
    #   corresponding (protein family, SMILES) keys and collect their PDB IDs,
    #   creating a list of PDB IDs for each set.

    # 4. Create SMILES sets for each split:

    #   a. Extract unique SMILES from the (protein family, SMILES) keys for each
    #   set.

    # By following this approach, you ensure that PDB IDs and their associated
    # SMILES are moved together while keeping protein families and identical
    # SMILES from being split across the training, validation, and testing sets.

    return pdb_ids, all_smis


def reassign_overlapping_smiles(all_smis):
    # Reassign overlapping SMILES.

    # Find the overlaps (intersections) between pairs of sets.
    train_val = all_smis.train & all_smis.val
    val_test = all_smis.val & all_smis.test
    train_test = all_smis.train & all_smis.test

    # Find the SMILES that are in two sets but not in the third one
    train_val_not_test = train_val - all_smis.test
    val_test_not_train = val_test - all_smis.train
    train_test_not_val = train_test - all_smis.val

    # Find SMILES that are present in all three sets
    train_test_val = all_smis.train & all_smis.val & all_smis.test

    # Overlapping SMILES are reassigned to temporary sets
    a_train, a_val = _random_divide_two_prts(train_val_not_test)
    b_val, b_test = _random_divide_two_prts(val_test_not_train)
    c_train, c_test = _random_divide_two_prts(train_test_not_val)
    d_train, d_val, d_test = _random_divide_three_prts(train_test_val)

    # Update SMILES sets to include the reassigned SMILES and exclude the
    # overlapping ones
    all_smis.train = (
        (all_smis.train - (all_smis.val | all_smis.test)) | a_train | c_train | d_train
    )
    all_smis.val = (
        (all_smis.val - (all_smis.train | all_smis.test)) | a_val | b_val | d_val
    )
    all_smis.test = (
        (all_smis.test - (all_smis.train | all_smis.val)) | b_test | c_test | d_test
    )


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
    pdb_ids = _limit_split_size(max_pdbs_train, max_pdbs_val, max_pdbs_test, pdb_ids,)

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
            "val": {"pdbs": len(set(pdb_ids.val)), "frags": len(set(all_smis.val)),},
            "test": {"pdbs": len(set(pdb_ids.test)), "frags": len(set(all_smis.test)),},
        },
        "train": {"pdbs": pdb_ids.train, "smiles": [smi for smi in all_smis.train],},
        "val": {"pdbs": pdb_ids.val, "smiles": [smi for smi in all_smis.val]},
        "test": {"pdbs": pdb_ids.test, "smiles": [smi for smi in all_smis.test]},
    }
    if not os.path.exists(os.path.dirname(save_splits)):
        os.mkdir(os.path.dirname(save_splits))
    with open(save_splits, "w") as f:
        json.dump(split_inf, f, indent=4)


def compute_dataset_split(
    dataset: "MOADInterface",
    seed: int = 0,
    fraction_train: float = 0.6,
    fraction_val: float = 0.5,
    prevent_smiles_overlap: bool = True,
    save_splits: str = None,
    load_splits: str = None,
    max_pdbs_train: int = None,
    max_pdbs_val: int = None,
    max_pdbs_test: int = None,
    butina_cluster_cutoff=0.4,
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
        # Not loading previously determined splits from disk, so generate based
        # on random seed.
        pdb_ids, all_smis = _generate_splits_from_scratch(
            dataset,
            fraction_train,
            fraction_val,
            prevent_smiles_overlap,
            max_pdbs_train,
            max_pdbs_val,
            max_pdbs_test,
            butina_cluster_cutoff,
        )
    else:
        # User has asked to load splits from file on disk. Get from the file.
        pdb_ids, all_smis, seed = _load_splits_from_disk(
            dataset, load_splits, max_pdbs_train, max_pdbs_val, max_pdbs_test,
        )

    if save_splits is not None:
        # Save spits and seed to json (for record keeping).
        _save_split(save_splits, seed, pdb_ids, all_smis)

    print(f"Training dataset size: {len(pdb_ids.train)}")
    print(f"Validation dataset size: {len(pdb_ids.val)}")
    print(f"Test dataset size: {len(pdb_ids.test)}")

    return (
        MOAD_split(name="TRAIN", targets=pdb_ids.train, smiles=all_smis.train),
        MOAD_split(name="VAL", targets=pdb_ids.val, smiles=all_smis.val),
        MOAD_split(name="TEST", targets=pdb_ids.test, smiles=all_smis.test),
    )


def full_moad_split(moad: "MOADInterface") -> MOAD_split:
    """Return a split containing all targets and smiles strings."""
    pdb_ids, all_smis = _generate_splits_from_scratch(
        moad,
        fraction_train=1.0,
        fraction_val=0.0,
        prevent_smiles_overlap=True,
        max_pdbs_train=None,
        max_pdbs_val=None,
        max_pdbs_test=None,
        butina_cluster_cutoff=0.0,
    )

    return MOAD_split(name="Full", targets=pdb_ids.train, smiles=all_smis.train)
