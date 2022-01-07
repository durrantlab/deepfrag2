
from torch import nn


_cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def cos_loss(yp, yt):
    """Cosine distance as a loss (inverted)."""
    return 1 - _cos(yp, yt)
