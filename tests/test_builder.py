import pytest
import torch
from torch import nn

from torchic.nn import NeuralNetwork
from torchic.nn.builder import NeuralNetworkBuilder
from torchic.utils import DEVICE


@pytest.fixture
def builder() -> NeuralNetworkBuilder:
    return NeuralNetworkBuilder(device=torch.device(DEVICE))


class TestNeuralNetworkBuilder:

    def test_simple_linear_relu(self, builder: NeuralNetworkBuilder) -> None:
        # Build using standard PyTorch
        expected: nn.Module = nn.ModuleDict(
            {
                "linear1": nn.Linear(10, 20),
                "relu": nn.ReLU(),
                "linear2": nn.Linear(20, 1),
            }
        )

        # Build using NeuralNetworkBuilder
        actual: NeuralNetwork = (
            builder.add_layer("linear1", nn.Linear(10, 20))
            .add_layer("relu", nn.ReLU())
            .add_layer("linear2", nn.Linear(20, 1))
            .build()
        )
        self.__compare_model(expected, actual.layers)

    def __compare_model(self, expected: nn.Module, actual: nn.Module) -> None:
        assert repr(expected) == repr(actual)
        assert str(expected) == str(actual)
        for layer_expected, layer_actual in zip(expected.modules(), actual.modules()):
            assert isinstance(layer_actual, type(layer_expected))
            if isinstance(layer_expected, nn.Linear) and isinstance(
                layer_actual, nn.Linear
            ):
                assert layer_expected.in_features == layer_actual.in_features
                assert layer_expected.out_features == layer_actual.out_features
