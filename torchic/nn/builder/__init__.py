from copy import deepcopy
from typing import Dict, Callable

import torch
import torch.nn as nn
from torch import Tensor

from torchic.nn import NeuralNetwork
from torchic.nn.layers import TransformLayer


class NeuralNetworkBuilder:
    """
    Builder for creating NeuralNetwork instances, supports nested layers.
    """

    def __init__(self, device: torch.device) -> None:
        """
        Initializes the NeuralNetworkBuilder with an input size.
        :param device: the device to run the network on
        """
        self.layers: nn.ModuleDict = nn.ModuleDict()
        self.device: torch.device = device

    def add_layer(self, name: str, layer: nn.Module) -> "NeuralNetworkBuilder":
        """
        Add a layer to the network.
        :param name: layer name
        :param layer: nn.Module layer
        :return: self
        """
        if name in self.layers:
            raise ValueError(f"Layer name '{name}' already exists.")
        self.layers[name] = layer
        return self

    def add_parallel(
        self, name: str, layer_dict: Dict[str, nn.Module]
    ) -> "NeuralNetworkBuilder":
        """
        Add parallel layers as a single block.
        :param name: layer name
        :param layer_dict: dictionary of nn.Module layers
        :return: self
        """
        if name in self.layers:
            raise ValueError(f"Layer name '{name}' already exists.")
        self.layers[name] = nn.ModuleDict(layer_dict)
        return self

    def add_transform_layer(
        self, name: str, transform_fn: Callable[[Tensor], Tensor]
    ) -> "NeuralNetworkBuilder":
        """
        Add a transformation layer to the network.
        It transforms the input tensor using the specified function.
        :param name: layer name
        :param transform_fn: transformation function
        :return: self
        """
        if name in self.layers:
            raise ValueError(f"Layer name '{name}' already exists.")
        self.layers[name] = TransformLayer(transform_fn)
        return self

    def build(self) -> NeuralNetwork:
        """
        Build the NeuralNetwork instance.
        :return: the neural network created up to this point
        """
        if len(self.layers) == 0:
            raise ValueError("No layers have been added to the network.")
        layers: nn.ModuleDict = deepcopy(self.layers)
        self.layers.clear()
        return NeuralNetwork(layers).to(self.device)
