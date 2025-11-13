import warnings
from abc import ABC, abstractmethod
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torchic.nn import NeuralNetwork
from torchic.utils import logger


class AbstractTrainer(ABC):
    """
    Abstract base class for neural network trainers.

    This class provides a framework for training neural networks with customizable
    training and evaluation steps. It handles the training loop, loss tracking,
    and basic logging functionality.

    Attributes:
        model (NeuralNetwork): The neural network model to be trained.
    """

    def __init__(self, model: NeuralNetwork) -> None:
        warnings.warn(
            "AbstractTrainer is not meant to be used for now. "
            "Please avoid using it.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()
        self.model: NeuralNetwork = model

    def fit(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer,
        epochs: int = 5,
    ) -> None:
        """
        Train the model using the provided data and parameters.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            test_dataloader (DataLoader): DataLoader for validation/test data.
            loss_fn (Callable[[Tensor, Tensor], Tensor]): Loss function to compute model error.
            optimizer (Optimizer): Optimizer for updating model parameters.
            epochs (int, optional): Number of training epochs. Defaults to 5.
        """
        self.model.train_losses.clear()
        self.model.test_losses.clear()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.__train(train_dataloader, loss_fn, optimizer, epoch)
            self.__test(test_dataloader, loss_fn, epoch)
        print("Done!")

    def __train(
        self,
        dataloader,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer,
        epoch: int,
    ) -> None:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        batch_loss: float = 0.0
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            input_batch, target = X.to(self.model.device()), y.to(self.model.device())
            pred, loss = self.train_step(input_batch, target, loss_fn, epoch)
            batch_loss += loss
            optimizer.step()
            optimizer.zero_grad()
            if batch % 100 == 0:
                loss_value, current = loss, (batch + 1) * len(input_batch)
                print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

        avg_loss: float = batch_loss / num_batches
        self.model.train_losses.append(avg_loss)

    def __test(
        self, dataloader, loss_fn: Callable[[Tensor, Tensor], Tensor], epoch: int
    ) -> None:
        size: int = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss: float = 0
        correct_predictions: int = 0
        with torch.no_grad():
            for X, y in dataloader:
                input_batch, target = (
                    X.to(self.model.device()),
                    y.to(self.model.device()),
                )
                pred, loss = self.eval_step(input_batch, target, loss_fn, epoch)
                test_loss = test_loss + loss
                predictions_tensor: Tensor = pred.argmax(1) == target
                correct_predictions = correct_predictions + int(
                    predictions_tensor.int().sum().item()
                )

        avg_loss: float = test_loss / num_batches
        self.model.test_losses.append(avg_loss)
        accuracy: float = 100 * correct_predictions / size
        print(
            f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {avg_loss:>8f} \n"
        )

    @abstractmethod
    def train_step(
        self, batch: Tensor, target: Tensor, loss_fn: Callable, epoch: int
    ) -> Tuple[Tensor, torch.types.Number]:
        """
        Perform a single training step.
        It must include loss backpropagation.

        Args:
            batch (Tensor): Input data batch.
            target (Tensor): Target values for the batch.
            loss_fn (Callable): Loss function to compute the error.

        Returns:
            Tuple[Tensor, torch.types.Number]: Tuple containing model predictions and loss value.
        """
        pass

    @abstractmethod
    def eval_step(
        self, batch: Tensor, target: Tensor, loss_fn: Callable, epoch: int
    ) -> Tuple[Tensor, torch.types.Number]:
        """
        Perform a single evaluation step.

        Args:
            batch (Tensor): Input data batch.
            target (Tensor): Target values for the batch.
            loss_fn (Callable): Loss function to compute the error.

        Returns:
            Tuple[Tensor, torch.types.Number]: Tuple containing model predictions and loss value.
        """
        pass


class DefaultTrainer(AbstractTrainer):
    """
    Default implementation of the AbstractTrainer.

    This trainer implements standard training and evaluation steps for neural networks,
    including forward pass, loss computation, and backpropagation.
    """

    def __init__(self, model: NeuralNetwork) -> None:
        warnings.warn(
            "DefaultTrainer is not meant to be used for now. "
            "Please avoid using it.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(model)

    def train_step(
        self, input_batch: Tensor, target: Tensor, loss_fn: Callable, epoch: int
    ) -> Tuple[Tensor, torch.types.Number]:
        # Compute prediction error
        pred: Tensor = self.model(input_batch)
        loss: Tensor = loss_fn(pred, target)
        logger.info(f"loss: {loss}")
        # Backpropagation
        # if loss is not reduced to a scalar
        if loss.dim() != 0:
            loss = loss.mean()

        loss.backward()
        return pred, loss.item()

    def eval_step(
        self, input_batch: Tensor, target: Tensor, loss_fn: Callable, epoch: int
    ) -> Tuple[Tensor, torch.types.Number]:
        pred: Tensor = self.model(input_batch)
        loss: Tensor = loss_fn(pred, target)
        if loss.dim() != 0:
            loss = loss.mean()
        return pred, loss.item()
