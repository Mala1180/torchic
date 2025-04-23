from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader


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
            result: Tensor = self(
                input_data.to(self.device())
            )  # Use model(input) to ensure proper behavior
        return InferenceResult(result)

    def fit(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        epochs: int = 5,
    ) -> None:
        self.train_losses.clear()
        self.test_losses.clear()
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.__train(train_dataloader, loss_fn, optimizer)
            self.__test(test_dataloader, loss_fn)
        print("Done!")

    def __train(self, dataloader, loss_fn, optimizer) -> None:
        size = len(dataloader.dataset)
        self.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device()), y.to(self.device())

            # Compute prediction error
            pred: Tensor = self(X)
            loss: Tensor = loss_fn(pred, y)
            # Backpropagation
            # if loss is not reduced to a scalar
            if loss.dim() != 0:
                loss = loss.mean()
            self.train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 100 == 0:
                loss_value, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

    def __test(self, dataloader, loss_fn) -> None:
        size: int = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        test_loss: float = 0
        correct_predictions: int = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device()), y.to(self.device())
                pred: Tensor = self(X)
                loss: Tensor = loss_fn(pred, y)
                if loss.dim() != 0:
                    loss = loss.mean()
                self.test_losses.append(loss.item())
                test_loss = test_loss + loss.item()
                predictions_tensor: Tensor = pred.argmax(1) == y
                correct_predictions = correct_predictions + int(
                    predictions_tensor.int().sum().item()
                )

        avg_loss: float = test_loss / num_batches
        accuracy: float = 100 * correct_predictions / size
        print(
            f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {avg_loss:>8f} \n"
        )

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
