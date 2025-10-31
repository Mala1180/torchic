import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from torchic.nn.builder import NeuralNetworkBuilder
from torchic.nn.trainers import DefaultTrainer
from torchic.utils import get_current_device

DEVICE = get_current_device()


class TestDefaultTrainer:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Setup shared resources
        self.model = (
            NeuralNetworkBuilder(DEVICE)
            .add_linear(10, 20)
            .add_layer(nn.ReLU())
            .add_linear(20, 2)
            .build()
        )
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=0.01)

        X = torch.randn(100, 10).to(DEVICE)
        y = torch.randint(0, 2, (100,)).to(DEVICE)
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=10)

        self.trainer = DefaultTrainer(self.model)

    def test_fit_runs_and_accumulates_losses(self):
        self.trainer.fit(
            train_dataloader=self.dataloader,
            test_dataloader=self.dataloader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            epochs=10,
        )

        assert len(self.model.train_losses) > 0, "Train losses should be collected."
        assert len(self.model.test_losses) > 0, "Test losses should be collected."

    def test_train_step_returns_valid_output(self):
        batch = torch.randn(10, 10).to(DEVICE)
        target = torch.randint(0, 2, (10,)).to(DEVICE)
        pred, loss = self.trainer.train_step(batch, target, self.loss_fn, 0)

        assert isinstance(pred, torch.Tensor)
        assert isinstance(loss, float)

    def test_eval_step_returns_valid_output(self):
        batch = torch.randn(10, 10).to(DEVICE)
        target = torch.randint(0, 2, (10,)).to(DEVICE)
        pred, loss = self.trainer.eval_step(batch, target, self.loss_fn, 0)

        assert isinstance(pred, torch.Tensor)
        assert isinstance(loss, float)
