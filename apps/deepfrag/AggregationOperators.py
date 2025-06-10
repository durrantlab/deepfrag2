"""Module for aggregation operators."""

from typing import Any, List, Union
import torch  # type: ignore
from torch import Tensor  # type: ignore
from torch.nn.modules.pooling import AdaptiveAvgPool3d  # type: ignore
from Fancy_aggregations import owas  # type: ignore
from Fancy_aggregations import integrals  # type: ignore
from enum import Enum
import numpy as np  # type: ignore


class Operator(Enum):

    """Enum for aggregation operators."""

    MEAN = "mean"
    OWA1 = "owa1"
    OWA2 = "owa2"
    OWA3 = "owa3"
    OWA_ExpSmooth1_01 = "owa_exp_smooth1_01"
    OWA_ExpSmooth1_02 = "owa_exp_smooth1_02"
    OWA_ExpSmooth1_03 = "owa_exp_smooth1_03"
    OWA_ExpSmooth1_04 = "owa_exp_smooth1_04"
    OWA_ExpSmooth1_05 = "owa_exp_smooth1_05"
    OWA_ExpSmooth1_06 = "owa_exp_smooth1_06"
    OWA_ExpSmooth1_07 = "owa_exp_smooth1_07"
    OWA_ExpSmooth1_08 = "owa_exp_smooth1_08"
    OWA_ExpSmooth1_09 = "owa_exp_smooth1_09"
    OWA_ExpSmooth2_01 = "owa_exp_smooth2_01"
    OWA_ExpSmooth2_02 = "owa_exp_smooth2_02"
    OWA_ExpSmooth2_03 = "owa_exp_smooth2_03"
    OWA_ExpSmooth2_04 = "owa_exp_smooth2_04"
    OWA_ExpSmooth2_05 = "owa_exp_smooth2_05"
    OWA_ExpSmooth2_06 = "owa_exp_smooth2_06"
    OWA_ExpSmooth2_07 = "owa_exp_smooth2_07"
    OWA_ExpSmooth2_08 = "owa_exp_smooth2_08"
    OWA_ExpSmooth2_09 = "owa_exp_smooth2_09"
    CHOQUET_CF = "choquet_integral_cf"
    CHOQUET_SYM = "choquet_integral_symmetric"
    SUGENO = "sugeno_fuzzy_integral"

    def startswith(self, prefix: str) -> bool:
        """
        Check if the value of the enum starts with a given prefix.
        
        Args:
            prefix (str): the prefix to check.
            
        Returns:
            bool: whether the value starts with the prefix.
        """
        return self.value.startswith(prefix)

    def rsplit(self, s: str) -> List[str]:
        """
        Split the value of the enum by a given separator.

        Args:
            s (str): the separator.

        Returns:
            List[str]: the split value.
        """
        return self.value.rsplit(s)


class Aggregate1DTensor:

    """Class for aggregating 1D tensors."""

    def __init__(self, operator: Operator):
        """Initialize an aggregator for 1D tensors.
        
        Args:
            operator (Operator): the aggregation operator to use.
        """
        self.function = None
        self.weight_function = None
        self.operator = operator

        # Below will be overwritten.
        self.weight_function_alpha: float = 0.0

        if self.operator == Operator.OWA1.value:
            self.function = owas.OWA1
        elif self.operator == Operator.OWA2.value:
            self.function = owas.OWA2
        elif self.operator == Operator.OWA3.value:
            self.function = owas.OWA3
        elif self.operator.startswith("owa_exp_smooth1"):
            self.function = owas.owa
            self.weight_function = self._exponential_smoothing_weights_1
            self.weight_function_alpha = float(
                "0." + self.operator.rsplit("owa_exp_smooth1_0")[1]
            )
        elif self.operator.startswith("owa_exp_smooth2"):
            self.function = owas.owa
            self.weight_function = self._exponential_smoothing_weights_2
            self.weight_function_alpha = float(
                "0." + self.operator.rsplit("owa_exp_smooth2_0")[1]
            )
        elif self.operator == Operator.CHOQUET_CF.value:
            self.function = integrals.choquet_integral_CF
        elif self.operator == Operator.CHOQUET_SYM.value:
            self.function = integrals.choquet_integral_symmetric
        elif self.operator == Operator.SUGENO.value:
            self.function = integrals.sugeno_fuzzy_integral
        elif self.operator != Operator.MEAN.value:
            raise ValueError("Aggregation operator is not valid")

    def aggregate_on_pytorch_tensor(self, tensor: Tensor) -> Union[Tensor, None]:
        """Aggregate a 1D tensor using the aggregation operator specified in
        the constructor.
        
        Args:
            tensor (Tensor): the tensor to aggregate.

        Returns:
            Union[Tensor, None]: the aggregated tensor.
        """
        if self.operator == Operator.MEAN.value:
            return tensor.mean()
        elif self.function is not None:
            if self.weight_function is None:
                return torch.tensor(
                    self.function(tensor.cpu().detach().numpy())[0],
                    dtype=torch.float32,
                    requires_grad=True,
                )

            numpy_array = tensor.cpu().detach().numpy()
            value = self.function(
                numpy_array,
                self.weight_function(len(numpy_array), self.weight_function_alpha),
            )
            return torch.tensor(value, dtype=torch.float32, requires_grad=True)
        return None

    def aggregate_on_numpy_array(self, numpy_array: np.ndarray) -> Union[float, None]:
        """Aggregate a 1D numpy array using the aggregation operator specified in
        the constructor.

        Args:
            numpy_array (np.ndarray): the array to aggregate.

        Returns:
            Union[float, None]: the aggregated value.
        """
        if self.operator == Operator.MEAN.value:
            return np.average(numpy_array)
        elif self.function is not None:
            if self.weight_function is None:
                return self.function(numpy_array)[0]
            else:
                return self.function(
                    numpy_array,
                    self.weight_function(len(numpy_array), self.weight_function_alpha),
                )
        return None

    @staticmethod
    def _exponential_smoothing_weights_1(dim: int, alpha: float) -> np.ndarray:
        """Compute the weights for the exponential smoothing aggregation operator
        with the first formulation.

        Args:
            dim (int): the dimension of the weight vector.
            alpha (float): the alpha value.

        Raises:
            Exception: if the dimension is less than 0 or the alpha value is not
            between 0 and 1.

        Returns:
            np.ndarray: the weight vector.
        """
        if dim <= 0:
            raise ValueError(
                "The dimension of the weight vector must be greater than 0"
            )
        if alpha < 0 or alpha > 1:
            raise ValueError("The alpha value must be between 0 and 1")

        weights = np.zeros(dim)
        weights[0] = alpha
        for i in range(2, dim):
            weights[i - 1] = weights[i - 2] * (1 - alpha)
        weights[dim - 1] = (1.0 - alpha) ** (dim - 1)

        return weights

    @staticmethod
    def _exponential_smoothing_weights_2(dim: int, alpha: float) -> np.ndarray:
        """Compute the weights for the exponential smoothing aggregation operator
        with the second formulation.

        Args:
            dim (int): the dimension of the weight vector.
            alpha (float): the alpha value.

        Raises:
            Exception: if the dimension is less than 0 or the alpha value is not
            between 0 and 1.

        Returns:
            np.ndarray: the weight vector.
        """
        if dim <= 0:
            raise ValueError(
                "The dimension of the weight vector must be greater than 0"
            )
        if alpha < 0 or alpha > 1:
            raise ValueError("The alpha value must be between 0 and 1")

        weights = np.zeros(dim)
        weights[dim - 1] = 1.0 - alpha
        for i in range(dim - 1, 1, -1):
            weights[i - 1] = weights[i] * (1 - weights[dim - 1])
        weights[0] = alpha ** (dim - 1)

        return weights


class Aggregate3x3Patches(Aggregate1DTensor, AdaptiveAvgPool3d):

    """Aggregate a 3D tensor using the aggregation operator specified in
    the constructor.
    """

    def __init__(self, operator: Operator, output_size: Any):
        """Initialize the class.
        
        Args:
            operator (Operator): the aggregation operator.
            output_size (Any): the output size.
        """
        Aggregate1DTensor.__init__(self, operator)
        AdaptiveAvgPool3d.__init__(self, output_size)

    def forward(self, tensor: Tensor) -> Tensor:
        """Aggregate a tensor using the aggregation operator specified in
        the constructor.

        Args:
            tensor (Tensor): the tensor to aggregate.

        Returns:
            Tensor: the aggregated tensor.
        """
        if self.operator == Operator.MEAN.value:
            return AdaptiveAvgPool3d.forward(self, tensor)

        tensor_resp = np.zeros(shape=(len(tensor), len(tensor[0]), 1, 1, 1))
        idx_patch = 0
        for patch in tensor:
            idx_channel = 0
            for channel in patch:
                values_in_matrix3d = [
                    value.item() for value in np.nditer(channel.cpu().detach().numpy())
                ]
                tensor_resp[idx_patch][idx_channel] = self.aggregate_on_numpy_array(
                    np.array(values_in_matrix3d)
                )
                idx_channel = idx_channel + 1
            idx_patch = idx_patch + 1

        # Only to check if the same result is obtained when using the mean operator (original)
        # original_via = AdaptiveAvgPool3d.forward(self, tensor)
        # alternat_via = torch.tensor(tensor_resp, dtype=torch.float32, requires_grad=True)

        return torch.tensor(tensor_resp, dtype=torch.float32, requires_grad=True)
