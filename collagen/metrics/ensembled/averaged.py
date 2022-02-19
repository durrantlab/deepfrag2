# ensemble averaged

from typing import Any
import torch


def create_initial_prediction_tensor(model_after_first_rotation: Any, num_rotations: int, device: Any) -> torch.Tensor:
    return model_after_first_rotation.predictions.detach().clone()

def udpate_prediction_tensor(existing_tensor: torch.Tensor, data_to_add: torch.Tensor, idx: int):
    torch.add(existing_tensor, data_to_add, out=existing_tensor)

def finalize_prediction_tensor(final_tensor: torch.Tensor, num_rotations: int, device: Any):
    # Divide by number of rotations to get the final average predicitons.
    torch.div(
        final_tensor, 
        torch.tensor(num_rotations, device=device),
        out=final_tensor
    )
    return None