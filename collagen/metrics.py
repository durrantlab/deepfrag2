
import torch
from torch import nn
from typing import List, Dict, Tuple
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import numpy as np

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
        predicton_targets: NxF tensor containing correct fingerprints.
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

        # The rank is the number of label-set distances that are equal to or
        # better (less) than the distance to the correct answer. TODO: Harrison:
        # Can you confirm change below ok?
        ranks[i] = torch.sum(dists <= d_target)
        # rank[i] = torch.sum(d_target <= dist)
    
    # TODO: Harrison: Can you confirm below is correct?
    return {v: torch.mean((ranks <= v).float()) for v in k}
    # return {v: torch.mean((ranks < v).float()) for v in k}

def most_similar_matches(predictions: torch.Tensor, label_set_fingerprints: torch.Tensor, k: int, ignore_duplicates=False)-> Tuple[torch.Tensor, torch.Tensor]:
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
    
    all_sorted_idxs = torch.zeros((len(predictions), k), dtype=torch.long)
    all_dists = torch.zeros((len(predictions), k))

    for i in tqdm(range(len(predictions)), desc='Most Similar Matches'):
        dists = _broadcast_fn(cos_loss, predictions[i], label_set_fingerprints)
        sorted_idxs = torch.argsort(dists, dim=-1).narrow(0, 0, k)
        all_sorted_idxs[i] = sorted_idxs
        all_dists[i] = torch.index_select(dists, 0, sorted_idxs)

    return all_sorted_idxs, all_dists

def project_predictions_onto_label_set_pca_space(
    predictions: torch.Tensor, label_set_fingerprints: torch.Tensor, 
    n_components: int
) -> List[torch.Tensor]:
    """
    Project the predictions onto the pca space defined by the label-set fingerprints.

    Args:
        predictions: SxNxF tensor containing predicted fingerprints. S is the number 
            of rotations, N is the number of predictions, F is the length of the 
            fingerprint.
        label_set_fingerprints: DxF tensor containing a fingerprint set.
        n_components (int): The number of PCA components to consider.
    """

    # Get all labelset fingerprints, but normalized.
    lblst_data_nmpy = label_set_fingerprints.cpu().numpy()
    transformer = Normalizer().fit(lblst_data_nmpy)
    transformer.transform(lblst_data_nmpy, copy=False)

    pca = PCA(n_components=n_components)
    pca.fit(lblst_data_nmpy)
    all_rotations_onto_pca = []
    for idx in tqdm(predictions.shape[1], desc='PCA Projections'):
        rotations_onto_pca = pca.transform(
            transformer.transform(
                predictions[:,idx].cpu().numpy()
            )
        )
        all_rotations_onto_pca.append(rotations_onto_pca)
    return all_rotations_onto_pca
