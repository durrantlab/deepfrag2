"""Functions and classes for assessing model predictions and performance."""

from dataclasses import dataclass
import torch  # type: ignore
from torch import nn  # type: ignore
from typing import Callable, List, Dict, Optional, Tuple, Any, TypeVar
from tqdm.auto import tqdm  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import Normalizer  # type: ignore
import numpy as np  # type: ignore


# Closer to 1 means more similar, closer to 0 means more dissimilar.
_cos = nn.CosineSimilarity(dim=1, eps=1e-6)
_mse = nn.MSELoss()


@dataclass
class PCAProject(object):

    """Sometimes it's helpful to project a high-dimensional fingerprint onto a
    lower-dimensional space (for visualization and comparison). This dataclass
    facilitates that projection process.
    """

    # TODO: How long does this take? Might not be worth calculating this during
    # testing if it takes a long time. Not really used much anymore.

    pca: Any
    transformer: Any

    def project(self, fingerprints: torch.Tensor) -> List[float]:
        """Project a fingerprint onto the PCA space.

        Args:
            fingerprints (torch.Tensor): The fingerprint(s) to project.

        Returns:
            List[float]: The projected fingerprint(s).
        """
        np_arr = (
            np.array([fingerprints.cpu().numpy()])
            if len(fingerprints.shape) == 1
            else fingerprints.cpu().numpy()
        )
        return self.pca.transform(self.transformer.transform(np_arr)).tolist()


def cos_loss(yp: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
    """Cosine distance as a loss (inverted). Smaller means more similar.

    Args:
        yp (torch.Tensor): Predicted fingerprint.
        yt (torch.Tensor): Target fingerprint.

    Returns:
        torch.Tensor: The loss.
    """
    # Closer to 1 means more dissimilar, closer to 0 means more similar.

    return 1 - _cos(yp, yt)


def mse_loss(yp: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
    """
    Args:
        yp (torch.Tensor): Predicted fingerprint.
        yt (torch.Tensor): Target fingerprint.

    Returns:
        torch.Tensor: The loss.
    """
    # Closer to 0 means more similar.

    return _mse(yp, yt)


def _broadcast_fn(fn: Callable, yp: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
    """Broadcast a distance function.

    Args:
        fn (callable): The distance function.
        yp (torch.Tensor): Predicted fingerprint.
        yt (torch.Tensor): Target fingerprint.

    Returns:
        torch.Tensor: The distance.
    """
    yp_b, yt_b = torch.broadcast_tensors(yp, yt)
    return fn(yp_b, yt_b)


def top_k(
    predictions: torch.Tensor,
    correct_predicton_targets: torch.Tensor,
    label_set_fingerprints: torch.Tensor,
    k: List[int],
    ignore_duplicates: bool = False,
) -> Dict[int, float]:
    """
    Batched Top-K accuracy.

    Args:
        predictions: NxF tensor containing predicted fingerprints.
        correct_predicton_targets: NxF tensor containing correct fingerprints.
        label_set_fingerprints: DxF tensor containing a fingerprint set.
        k (List[int]): K values to consider.
        ignore_duplicates (bool): If True, ignore duplicate fingerprints when ranking.

    Returns:
        Dict[int, float]: The top-k accuracies.
    """
    if ignore_duplicates:
        label_set_fingerprints = label_set_fingerprints.unique(dim=0)

    ranks = torch.zeros(len(predictions), dtype=torch.long)

    for i in tqdm(range(len(predictions)), desc="Top-K"):
        # Distances between this prediction and each of the label-set
        # fingerprints. Note cos_loss is the cosine distance (so smaller means
        # more similar).
        dists = _broadcast_fn(cos_loss, predictions[i], label_set_fingerprints)

        # The distance from this prediction and the correct answer. Note that
        # the correct answer must be among the answers in the label set.
        dist_to_target = cos_loss(
            predictions[i].unsqueeze(0), correct_predicton_targets[i].unsqueeze(0)
        )

        # Though the correct answer must be in the label set for top k to work,
        # at times it differs slightly, presumably due to rounding errors. So we
        # need to find the entry in dists that is closest to d_target (off by at
        # most only a tiny amount).
        min_idx = dists.sub(dist_to_target).abs().argmin()
        dist_to_target = dists[min_idx]

        # The rank is the number of label-set distances that are better (less)
        # than the distance to the correct answer.
        ranks[i] = torch.sum(dists < dist_to_target)

    # Rank is 0-indexed, K is 1-indexed
    # I.e. top-1 means frequency of rank 0
    #      top-5 means frequency of rank 0,1,2,3,4
    #      etc...
    return {v: torch.mean((ranks < v).float()) for v in k}


# TODO: Label set could be in a single class that includes both fingerprints and
# vectors, etc. Would be slick.

T = TypeVar("T")


def most_similar_matches(
    predictions: torch.Tensor,
    label_set_fingerprints: torch.Tensor,
    label_set_identifiers: List[T],
    k: int,
    pca_project: Optional[PCAProject] = None,
    ignore_duplicates=False,
) -> List[List[Tuple[T, float, Optional[List[float]]]]]:
    """Identify most similar entires in fingerprint library.

    Args:
        predictions: NxF tensor containing predicted fingerprints.
        label_set_fingerprints: DxF tensor containing a fingerprint set.
        label_set_identifiers (List[T]): List of SMILES or Entry_infos for each
            fingerprint in the label set.
        k (int): Top K values to consider.
        pca_project (Optional[PCAProject]): PCA space to project fingerprints into.
        ignore_duplicates (bool): If True, ignore duplicate fingerprints.

    Returns:
        List[List[Tuple[str, float, Optional[List[float]]]]]: List of [SMILES,
            distance, projected fingerprint] for each entry.
    """
    if ignore_duplicates:
        label_set_fingerprints = label_set_fingerprints.unique(dim=0)

    all_most_similar: List[List[Tuple[T, float, Optional[List[float]]]]] = []

    if k > len(label_set_fingerprints):
        raise ValueError(
            f"Requested top-K ({k}) is greater than the number of label set fingerprints ({len(label_set_fingerprints)}). You may need to reduce your --num_inference_predictions parameter."
        )
    
    # If predictions[0] is on device cpu but label_set_fingerprints is on device cuda, then
    # move predictions to device cuda.
    if predictions.device != label_set_fingerprints.device:
        predictions = predictions.to(label_set_fingerprints.device)

    for entry_idx in tqdm(range(len(predictions)), desc="Most Similar Matches"):
        dists = _broadcast_fn(cos_loss, predictions[entry_idx], label_set_fingerprints)
        sorted_idxs = torch.argsort(dists, dim=-1).narrow(0, 0, k)
        sorted_dists: torch.Tensor = torch.index_select(dists, 0, sorted_idxs)
        sorted_smis: List[T] = [label_set_identifiers[idx] for idx in sorted_idxs]

        # if pca_project is not None:
        sorted_label_set_fingerprints = torch.index_select(
            label_set_fingerprints, 0, sorted_idxs
        )

        most_similar: List[Tuple[T, float, Optional[List[float]]]] = []

        for cos_dist, smi, fp in zip(
            sorted_dists[:k], sorted_smis[:k], sorted_label_set_fingerprints[:k]
        ):
            # So reports cos similarity, not cosine distance.
            cos_sim = 1 - float(cos_dist)
            if pca_project is not None:
                similar_one_to_add: Tuple[T, float, Optional[List[float]]] = (
                    smi,
                    cos_sim,
                    pca_project.project(fp),
                )
            else:
                similar_one_to_add: Tuple[T, float, Optional[List[float]]] = (
                    smi,
                    cos_sim,
                    None,
                )

            most_similar.append(similar_one_to_add)

        all_most_similar.append(most_similar)

    return all_most_similar


def pca_space_from_label_set_fingerprints(
    label_set_fingerprints: torch.Tensor, n_components: int
) -> PCAProject:
    """Create a PCA space from a set of fingerprints. Other fingerprints can be
    projected onto this space elsewhere.

    Args:
        label_set_fingerprints (torch.Tensor): DxF tensor containing a fingerprint set.
        n_components (int): Number of components to use in PCA.

    Returns:
        PCAProject: The PCA space.
    """
    # Get all labelset fingerprints, but normalized.
    lblst_data_nmpy = label_set_fingerprints.cpu().numpy()
    transformer = Normalizer().fit(lblst_data_nmpy)
    transformer.transform(lblst_data_nmpy, copy=False)

    # Create the label-set PCA (or other) space.
    pca = PCA(n_components=n_components)
    # pca = TSNE(n_components=n_components, learning_rate='auto', init='random', n_jobs=16)

    pca.fit(lblst_data_nmpy)

    return PCAProject(pca, transformer)
