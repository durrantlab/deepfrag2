
import torch

from .base_pytorch import BasePytorchModel
from ..molecule_util import MolGraph


class MolAutoencoder(BasePytorchModel):

    def encode(self, mol: MolGraph) -> torch.Tensor:
        raise NotImplementedError()
    
    def decode(self, z: torch.Tensor, template: MolGraph) -> MolGraph:
        raise NotImplementedError()
