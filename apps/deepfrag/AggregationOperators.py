import torch
from torch import Tensor
from torch.nn.modules.pooling import AdaptiveAvgPool3d
from Fancy_aggregations import owas
from Fancy_aggregations import integrals
from enum import Enum
import numpy as np


class Operator(Enum):
    MEAN = "mean"
    OWA1 = "owa1"
    OWA2 = "owa2"
    OWA3 = "owa3"
    CHOQUET_CF = "choquet_integral_CF"
    CHOQUET_SYM = "choquet_integral_symmetric"
    SUGENO = "sugeno_fuzzy_integral"


class Aggregate1DTensor:

    def __init__(self, operator: Operator):

        self.function = None
        self.operator = operator

        if self.operator.value == Operator.OWA1.value:
            self.function = owas.OWA1
        elif self.operator.value == Operator.OWA2.value:
            self.function = owas.OWA2
        elif self.operator.value == Operator.OWA3.value:
            self.function = owas.OWA3
        elif self.operator.value == Operator.CHOQUET_CF.value:
            self.function = integrals.choquet_integral_CF
        elif self.operator.value == Operator.CHOQUET_SYM.value:
            self.function = integrals.choquet_integral_symmetric
        elif self.operator.value == Operator.SUGENO.value:
            self.function = integrals.sugeno_fuzzy_integral

    def aggregate_on_pytorch_tensor(self, tensor: Tensor):
        # m1 = tensor.mean()
        # o1 = owas.OWA1(tensor.detach().numpy())
        # o2 = owas.OWA2(tensor.detach().numpy())
        # o3 = owas.OWA3(tensor.detach().numpy())
        # c1 = integrals.choquet_integral_CF(tensor.detach().numpy())
        # c2 = integrals.choquet_integral_symmetric(tensor.detach().numpy())
        # s1 = integrals.sugeno_fuzzy_integral(tensor.detach().numpy())
        if self.operator.value == Operator.MEAN.value:
            return tensor.mean()
        elif self.function is not None:
            return torch.tensor(self.function(tensor.detach().numpy())[0], dtype=torch.float32, requires_grad=True)
        return None

    def aggregate_on_numpy_array(self, numpy_array):
        if self.operator.value == Operator.MEAN.value:
            return np.average(numpy_array)
        elif self.function is not None:
            return self.function(numpy_array)[0]
        return None


class Aggregate3x3Patches(Aggregate1DTensor, AdaptiveAvgPool3d):

    def __init__(self, operator: Operator, output_size):
        Aggregate1DTensor.__init__(self, operator)
        AdaptiveAvgPool3d.__init__(self, output_size)

    def forward(self, tensor: Tensor) -> Tensor:
        if self.operator.value == Operator.MEAN.value:
            return AdaptiveAvgPool3d.forward(self, tensor)

        tensor_resp = np.zeros(shape=(len(tensor), len(tensor[0]), 1, 1, 1))
        idx_patch = 0
        for patch in tensor:
            idx_channel = 0
            for channel in patch:
                values_in_matrix3d = []
                for value in np.nditer(channel.detach().numpy()):
                    values_in_matrix3d.append(value.item())
                tensor_resp[idx_patch][idx_channel] = self.aggregate_on_numpy_array(np.array(values_in_matrix3d))
                idx_channel = idx_channel + 1
            idx_patch = idx_patch + 1

        # Only to check if the same result is obtained when using the mean operator (original)
        # original_via = AdaptiveAvgPool3d.forward(self, tensor)
        # alternat_via = torch.tensor(tensor_resp, dtype=torch.float32, requires_grad=True)

        return torch.tensor(tensor_resp, dtype=torch.float32, requires_grad=True)
