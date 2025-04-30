from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor


class InferenceResult:
    def __init__(self, result: Tensor) -> None:
        self.tensor = result
        self.predicted = self.tensor.squeeze().argmax()


class NeuralNetwork(nn.Module):
    def __init__(self, layers: nn.ModuleDict) -> None:
        super().__init__()
        self.layers: nn.ModuleDict = layers
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []

    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        for name, layer in self.layers.items():
            if isinstance(layer, nn.ModuleDict):
                inputs_number: int = x.size(0)
                tensors: Tuple[Tensor, ...] = torch.chunk(x, inputs_number, dim=0)
                inputs: List[Tensor] = [t.squeeze(dim=0) for t in tensors]
                outputs = [
                    sublayer(inputs[index])
                    for index, sublayer in enumerate(layer.values())
                ]
                x = torch.cat(outputs, dim=1)  # Concatenate along feature dimension
            else:
                x = layer(x)  # Sequential processing
        return x

    def inference(self, input_data: Tensor) -> InferenceResult:
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            result: Tensor = self(input_data.to(self.device()))
        return InferenceResult(result)

    def save(self, path: str) -> None:
        save_dict = {
            "model_state_dict": self.state_dict(),
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
        }
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        neural_network_model = torch.load(path, weights_only=True)
        self.load_state_dict(neural_network_model["model_state_dict"])
        self.train_losses = neural_network_model["train_losses"]
        self.test_losses = neural_network_model["test_losses"]

    def plot_loss(self) -> None:
        if not self.train_losses or not self.test_losses:
            print("The network has not been trained yet.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

        # Plot training loss on the first subplot
        ax1.plot(
            range(1, len(self.train_losses) + 1),
            self.train_losses,
            label="Training Loss",
            color="blue",
        )
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Batch")
        ax1.set_ylabel("Loss Value")

        # Plot testing loss on the second subplot
        ax2.plot(
            range(1, len(self.test_losses) + 1),
            self.test_losses,
            label="Testing Loss",
            color="red",
        )
        ax2.set_title("Testing Loss")
        ax2.set_xlabel("Batch")
        ax2.set_ylabel("Loss Value")

        plt.tight_layout()
        plt.show()
