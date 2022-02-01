
from dataclasses import dataclass
import torch
from torch import nn
from typing import List, Dict, Tuple, Any
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import numpy as np

@dataclass
class PCAProject(object):
    pca: Any
    transformer: Any

    def project(self, fingerprints: torch.Tensor) -> List[float]:
        if len(fingerprints.shape) == 1:
            np_arr = np.array([fingerprints.cpu().numpy()])
        else:
            np_arr = fingerprints.cpu().numpy()

        return self.pca.transform(self.transformer.transform(np_arr)).tolist()

_cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def cos_loss(yp, yt):
    """Cosine distance as a loss (inverted). Smaller means more similar."""
    return 1 - _cos(yp, yt)

def bin_acc(pred, target):
    """Binary accuracy."""
    return torch.mean((torch.round(pred) == target).float())


def _broadcast_fn(fn, yp, yt):
    """Broadcast a distance function."""
    yp_b, yt_b = torch.broadcast_tensors(yp, yt)
    return fn(yp_b, yt_b)


def top_k(predictions: torch.Tensor, correct_predicton_targets: torch.Tensor, label_set_fingerprints: torch.Tensor, k: List[int], ignore_duplicates=False) -> Dict[int, float]:
    """
    Batched Top-K accuracy.

    Args:
        predictions: NxF tensor containing predicted fingerprints.
        correct_predicton_targets: NxF tensor containing correct fingerprints.
        label_set_fingerprints: DxF tensor containing a fingerprint set.
        k (List[int]): K values to consider.
        ignore_duplicates (bool): If True, ignore duplicate fingerprints when ranking.
    """
    if ignore_duplicates:
        label_set_fingerprints = label_set_fingerprints.unique(dim=0)
    
    ranks = torch.zeros(len(predictions), dtype=torch.long)

    for i in tqdm(range(len(predictions)), desc='Top-K'):
        # Distances between this prediction and each of the label-set
        # fingerprints.
        dists = _broadcast_fn(cos_loss, predictions[i], label_set_fingerprints)
        
        # The distance from this prediction and the correct answer.
        d_target = cos_loss(
            predictions[i].unsqueeze(0),
            correct_predicton_targets[i].unsqueeze(0)
        )

        # print("");print(dists.sort().values); print(d_target)

        # The rank is the number of label-set distances that are better (less)
        # than the distance to the correct answer.
        ranks[i] = torch.sum(dists < d_target)
    
    # Rank is 0-indexed, K is 1-indexed
    # I.e. top-1 means frequency of rank 0
    #      top-5 means frequency of rank 0,1,2,3,4
    #      etc...
    return {v: torch.mean((ranks < v).float()) for v in k}

# TODO: Label set could be in a single class that includes both fingerprints and
# vectors, etc. Would be slick.
def most_similar_matches(
    predictions: torch.Tensor, label_set_fingerprints: torch.Tensor,
    label_set_smis: List[str], k: int, pca_project: PCAProject=None,
    ignore_duplicates=False
):  # -> List[List[str, float, List[float]]]:
    """
    Identify most similar entires in fingerprint library.

    Args:
        predictions: NxF tensor containing predicted fingerprints.
        label_set_fingerprints: DxF tensor containing a fingerprint set.
        k (int): Top K values to consider.
        ignore_duplicates (bool): If True, ignore duplicate fingerprints.
    """

    if ignore_duplicates:
        label_set_fingerprints = label_set_fingerprints.unique(dim=0)
    
    # all_sorted_idxs = torch.zeros((len(predictions), k), dtype=torch.long)
    # all_dists = torch.zeros((len(predictions), k))

    all_most_similar = []

    for entry_idx in tqdm(range(len(predictions)), desc='Most Similar Matches'):
        dists = _broadcast_fn(cos_loss, predictions[entry_idx], label_set_fingerprints)
        sorted_idxs = torch.argsort(dists, dim=-1).narrow(0, 0, k)
        sorted_dists = torch.index_select(dists, 0, sorted_idxs)
        sorted_smis = [label_set_smis[idx] for idx in sorted_idxs]
        
        if pca_project is not None:
            sorted_label_set_fingerprints = torch.index_select(
                label_set_fingerprints, 0, sorted_idxs
            )

        most_similar = []

        for d, s, fp in zip(
            sorted_dists[:k], sorted_smis[:k], sorted_label_set_fingerprints[:k]
        ):
            to_add = [s, float(d)]
            if pca_project is not None:
                to_add.append(pca_project.project(fp))
            most_similar.append(to_add)

        all_most_similar.append(most_similar)

    return all_most_similar
   
def make_pca_space_from_label_set_fingerprints(
    label_set_fingerprints: torch.Tensor, n_components: int
) -> PCAProject:
    # Get all labelset fingerprints, but normalized.
    lblst_data_nmpy = label_set_fingerprints.cpu().numpy()
    transformer = Normalizer().fit(lblst_data_nmpy)
    transformer.transform(lblst_data_nmpy, copy=False)

    # Create the label-set PCA space.
    pca = PCA(n_components=n_components)
    pca.fit(lblst_data_nmpy)

    return PCAProject(pca, transformer)

