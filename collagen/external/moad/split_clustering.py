from typing import Any, List, Tuple
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

# TODO: Name of this file isn't obvious. Good to rename.

def _cluster_fps(fps: List[List[float]], cutoff: float=0.2) -> Any:
    """Clusters fingerprints using the Butina algorithm.
    
    Args:
        fps (List[List[float]]): List of fingerprints.
        cutoff (float, optional): Cutoff for clustering. Defaults to 0.2.

    Returns:
        Any: Clusters.
    """

    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    return Butina.ClusterData(dists, nfps, cutoff, isDistData=True, reordering=True)


def generate_splits_from_clustering(
    moad: "MOADInterface",
    split_rand_num_gen: Any,
    fraction_train: float = 0.6,
    fraction_val: float = 0.5,
    butina_cluster_cutoff: float = 0.4,
) -> Tuple[List, List, List]:
    """Generates test/train/val splits given clustering.

    Args:
        moad (MOADInterface): MOADInterface object.
        split_rand_num_gen: Random number generator.
        fraction_train (float, optional): Fraction of training data. Defaults to 0.6.
        fraction_val (float, optional): Fraction of validation data. Defaults to 0.5.
        butina_cluster_cutoff (float, optional): Cutoff for clustering. Defaults to 0.4.

    Returns:
        Tuple[List, List, List]: Train, validation, and test splits.
    """

    ligands = []
    targets = []
    for c in moad.classes:
        for f in c.families:
            for x in f.targets:
                ligands.append(x.ligands[0].rdmol)
                targets.append([x.pdb_id])

    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048) for x in ligands]
    clusters = _cluster_fps(fps, cutoff=butina_cluster_cutoff)

    train_families = []
    val_families = []
    test_families = []
    for cluster in clusters:
        targets4cluster = [targets[pos] for pos in cluster]
        size = len(targets4cluster)
        split_rand_num_gen.shuffle(targets4cluster)
        aux_list = targets4cluster[: int(size * fraction_train)]
        train_families.extend(iter(aux_list))
        aux_list = targets4cluster[int(size * fraction_train) :]

        size = len(aux_list)
        split_rand_num_gen.shuffle(aux_list)
        aux_list_1 = aux_list[: int(size * fraction_val)]
        val_families.extend(iter(aux_list_1))
        aux_list_1 = aux_list[int(size * fraction_val) :]
        test_families.extend(iter(aux_list_1))
    return train_families, val_families, test_families
