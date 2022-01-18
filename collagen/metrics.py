
import torch
from torch import nn
from typing import List, Dict
from tqdm.auto import tqdm


_cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def cos_loss(yp, yt):
    """Cosine distance as a loss (inverted)."""
    return 1 - _cos(yp, yt)

def bin_acc(pred, target):
    """Binary accuracy."""
    return torch.mean((torch.round(pred) == target).float())


def _broadcast_fn(fn, yp, yt):
    """Broadcast a distance function."""
    yp_b, yt_b = torch.broadcast_tensors(yp, yt)
    return fn(yp_b, yt_b)


def top_k(predictions: torch.Tensor, targets: torch.Tensor, fingerprints: torch.Tensor, k: List[int], ignore_duplicates=True) -> Dict[int, float]:
    """
    Batched Top-K accuracy.

    Args:
        predictions: NxF tensor containing predicted fingerprints.
        targets: NxF tensor containing correct fingerprints.
        fingerprints: DxF tensor containing a fingerprint set.
        k (List[int]): K values to consider.
        ignore_duplicates (bool): If True, ignore duplicate fingerprints when ranking.
    """
    if ignore_duplicates:
        fingerprints = fingerprints.unique(dim=0)
    
    rank = torch.zeros(len(predictions), dtype=torch.long)

    for i in tqdm(range(len(predictions)), desc='Top-K'):
        dist = _broadcast_fn(cos_loss, predictions[i], fingerprints)
        d_target = cos_loss(predictions[i].unsqueeze(0), targets[i].unsqueeze(0))
        rank[i] = torch.sum(d_target <= dist)

    return {v: torch.mean((rank < v).float()) for v in k}
