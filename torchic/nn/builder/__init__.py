from typing import Dict, Callable, Optional, Set

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

    def add_layer(
        self, layer: nn.Module, name: Optional[str] = None
    ) -> "NeuralNetworkBuilder":
        """
        Add a generic layer to the network.
        :param name: Layer name
        :param layer: nn.Module layer
        :return: self
        """
        self.__assert_layer_name(name)
        if name is None:
            name = self.__provide_layer_name(layer.__class__.__name__.lower())
        self.layers[name] = layer
        return self

    def add_linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: Optional[str] = None,
    ) -> "NeuralNetworkBuilder":
        """
        Add a linear layer to the network.
        :param in_features: Number of input features
        :param out_features: Number of output features
        :param bias: Whether to include a bias term
        :param name: Layer name
        :return: self
        """
        self.__assert_layer_name(name)
        linear_layer = nn.Linear(in_features, out_features, bias)
        if name is None:
            name = self.__provide_layer_name("linear")
        self.layers[name] = linear_layer
        return self

    def add_parallel(
        self,
        layer_dict: Dict[str, nn.Module] | Set[nn.Module],
        name: Optional[str] = None,
    ) -> "NeuralNetworkBuilder":
        """
        Add parallel layers as a single block.
        :param name: Layer name
        :param layer_dict: Dictionary of nn.Module layers
        :return: self
        """
        self.__assert_layer_name(name)
        if name is None:
            name = self.__provide_layer_name("parallel")
        if isinstance(layer_dict, set):
            layer_dict = {
                self.__provide_layer_name(layer.__class__.__name__.lower()): layer
                for layer in layer_dict
            }
        elif isinstance(layer_dict, dict):
            pass
        else:
            raise TypeError(
                "layer_dict must be of type Dict[str, torch.nn.Module] or Set[nn.Module]."
            )
        self.layers[name] = nn.ModuleDict(layer_dict)
        return self

    def add_transform(
        self, transform_fn: Callable[[Tensor], Tensor], name: Optional[str] = None
    ) -> "NeuralNetworkBuilder":
        """
        Add a transformation layer to the network.
        It transforms the input tensor using the specified function.
        :param name: Layer name
        :param transform_fn: Transformation function
        :return: self
        """
        self.__assert_layer_name(name)
        if name is None:
            name = self.__provide_layer_name("transform")
        self.layers[name] = TransformLayer(transform_fn)
        return self

    def build(self) -> NeuralNetwork:
        """
        Build the NeuralNetwork instance.
        :return: The neural network created up to this point
        """
        if len(self.layers) == 0:
            raise ValueError("No layers have been added to the network.")
        return NeuralNetwork(self.layers).to(self.device)

    def reset(self) -> None:
        """
        Reset the builder to its initial state.
        :return: None
        """
        self.layers.clear()

    def __assert_layer_name(self, name: Optional[str]) -> None:
        if name is not None and name in self.layers:
            raise ValueError(f"Layer name '{name}' already exists.")

    def __provide_layer_name(self, base_name: str) -> str:
        """
        Provide a unique name for the layer.
        :param base_name: Base name for the layer
        :return: Unique layer name
        """
        counter: int = 1
        name: str = base_name + str(counter)
        while name in self.layers:
            name = base_name + str(counter)
            counter += 1
        return name
