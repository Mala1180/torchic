from typing import Callable

import torch.nn as nn
from torch import Tensor


class TransformLayer(nn.Module):
    """
    A layer that applies an arbitrary transformation to a tensor.
    """

    def __init__(self, transform_fn: Callable[[Tensor], Tensor]) -> None:
        """
        Initializes the TransformLayer with a transformation function.
        :param transform_fn: the transformation function
        """
        super().__init__()
        self.transform_function: Callable[[Tensor], Tensor] = transform_fn

    def forward(self, x: Tensor) -> Tensor:
        return self.transform_function(x)
