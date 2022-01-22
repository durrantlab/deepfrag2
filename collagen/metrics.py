
import torch
from torch import nn
from typing import List, Dict, Tuple
from tqdm.auto import tqdm


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


def top_k(predictions: torch.Tensor, predicton_targets: torch.Tensor, label_set_fingerprints: torch.Tensor, k: List[int], ignore_duplicates=False) -> Dict[int, float]:
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
    
    rank = torch.zeros(len(predictions), dtype=torch.long)

    for i in tqdm(range(len(predictions)), desc='Top-K'):
        dist = _broadcast_fn(cos_loss, predictions[i], label_set_fingerprints)
        
        # The distance to the correct answer.
        d_target = cos_loss(predictions[i].unsqueeze(0), predicton_targets[i].unsqueeze(0))
        
        # The rank is the number of fingerprints that are equal to or better
        # than the distance to the correct answer.
        rank[i] = torch.sum(d_target <= dist)

    return {v: torch.mean((rank < v).float()) for v in k}

def most_similar_matches(predictions: torch.Tensor, label_set_fingerprints: torch.Tensor, k: int, ignore_duplicates=False)-> Tuple[torch.Tensor, torch.Tensor]:
    """
    Identify most similar entires in fingerprint library.

    Args:
        predictions: NxF tensor containing predicted fingerprints.
        label_set_fingerprints: DxF tensor containing a fingerprint set.
        k (List[int]): Top K values to consider.
        ignore_duplicates (bool): If True, ignore duplicate fingerprints.
    """

    if ignore_duplicates:
        label_set_fingerprints = label_set_fingerprints.unique(dim=0)
    
    all_sorted_idxs = torch.zeros((len(predictions), k), dtype=torch.long)
    all_dists = torch.zeros((len(predictions), k))

    for i in tqdm(range(len(predictions)), desc='Top-K'):
        dists = _broadcast_fn(cos_loss, predictions[i], label_set_fingerprints)
        sorted_idxs = torch.argsort(dists, dim=-1).narrow(0, 0, k)
        all_sorted_idxs[i] = sorted_idxs
        all_dists[i] = torch.index_select(dists, 0, sorted_idxs)

    return all_sorted_idxs, all_dists