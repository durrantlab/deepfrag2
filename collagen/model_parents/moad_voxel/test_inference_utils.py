"""Testing and inference share a lot of code. Let's put the common code here."""

import torch  # type: ignore
from typing import Any, List, Tuple


def remove_redundant_fingerprints(
    label_set_fps: torch.Tensor,
    label_set_associated_data: List[Any],
    device: torch.device,
) -> Tuple[torch.Tensor, List[Any]]:
    """Given ordered lists of fingerprints and smiles strings, removes
    redundant fingerprints and smis while maintaining the consistent order
    between the two lists.

    Args:
        label_set_fps (torch.Tensor): A tensor with the fingerprints.
        label_set_associated_data (List[Any]): A list of the associatd smiles strings or entry infos.
        device (torch.device): The device.

    Returns:
        Tuple[torch.Tensor, List[Any]]: Same as input, but redundant
            fingerprints are removed.
    """
    # label_set_fps and inverse_indices are both torch.Tensor
    label_set_fps, inverse_indices = label_set_fps.unique(dim=0, return_inverse=True)

    label_set_associated_data = [
        inf[1]
        for inf in sorted(
            [
                (inverse_idx, label_set_associated_data[smi_idx])
                for inverse_idx, smi_idx in {
                    int(inverse_idx): data_idx
                    for data_idx, inverse_idx in enumerate(inverse_indices)
                }.items()
            ]
        )
    ]

    return label_set_fps, label_set_associated_data
