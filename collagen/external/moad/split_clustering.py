from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina


def _cluster_fps(fps, cutoff=0.2):
    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1-x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True, reordering=True)
    return cs


def generate_splits_from_clustering(moad: "MOADInterface", split_rand_num_gen, fraction_train: float = 0.6, fraction_val: float = 0.5, butina_cluster_cutoff: float = 0.4):
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
        aux_list = targets4cluster[:int(size * fraction_train)]
        for x in aux_list:
            train_families.append(x)
        aux_list = targets4cluster[int(size * fraction_train):]

        size = len(aux_list)
        split_rand_num_gen.shuffle(aux_list)
        aux_list_1 = aux_list[:int(size * fraction_val)]
        for x in aux_list_1:
            val_families.append(x)
        aux_list_1 = aux_list[int(size * fraction_val):]
        for x in aux_list_1:
            test_families.append(x)

    return train_families, val_families, test_families
