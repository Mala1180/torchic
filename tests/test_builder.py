import pytest
from torch import nn

from torchic.nn import NeuralNetwork
from torchic.nn.builder import NeuralNetworkBuilder
from torchic.utils import get_current_device


@pytest.fixture
def builder() -> NeuralNetworkBuilder:
    return NeuralNetworkBuilder(device=get_current_device())


class TestNeuralNetworkBuilder:
    def test_create_linear_layers(self, builder: NeuralNetworkBuilder) -> None:
        expected: nn.Module = nn.ModuleDict(
            {
                "linear1": nn.Linear(20, 30),
                "linear2": nn.Linear(20, 10),
                "linear3": nn.Linear(10, 1),
            }
        )
        actual: NeuralNetwork = (
            builder.add_linear(20, 30).add_linear(20, 10).add_linear(10, 1).build()
        )
        self.__compare_models(expected, actual.layers)

    def test_create_generic_layers(self, builder: NeuralNetworkBuilder) -> None:
        # Build using standard PyTorch
        expected: nn.Module = nn.ModuleDict(
            {
                "linear1": nn.Linear(10, 20),
                "relu1": nn.ReLU(),
                "tanh1": nn.Tanh(),
                "linear2": nn.Linear(20, 1),
            }
        )

        # Build using NeuralNetworkBuilder
        actual: NeuralNetwork = (
            builder.add_layer(nn.Linear(10, 20))
            .add_layer(nn.ReLU())
            .add_layer(nn.Tanh())
            .add_layer(nn.Linear(20, 1))
            .build()
        )
        self.__compare_models(expected, actual.layers)

    def test_create_parallel_layers(self, builder: NeuralNetworkBuilder) -> None:
        # Build using standard PyTorch
        expected: nn.Module = nn.ModuleDict(
            {
                "linear1": nn.Linear(10, 20),
                "relu1": nn.ReLU(),
                "parallel1": nn.ModuleDict(
                    {
                        "linear2": nn.Linear(20, 1),
                        "linear3": nn.Linear(20, 1),
                    }
                ),
            }
        )

        # Build using NeuralNetworkBuilder
        actual: NeuralNetwork = (
            builder.add_layer(nn.Linear(10, 20))
            .add_layer(nn.ReLU())
            .add_parallel({"linear2": nn.Linear(20, 1), "linear3": nn.Linear(20, 1)})
            .build()
        )
        self.__compare_models(expected, actual.layers)

    def __compare_models(self, expected: nn.Module, actual: nn.Module) -> None:
        assert repr(expected) == repr(actual)
        assert str(expected) == str(actual)
        for layer_expected, layer_actual in zip(expected.modules(), actual.modules()):
            assert isinstance(layer_actual, type(layer_expected))
            if isinstance(layer_expected, nn.Linear) and isinstance(
                layer_actual, nn.Linear
            ):
                assert layer_expected.in_features == layer_actual.in_features
                assert layer_expected.out_features == layer_actual.out_features
