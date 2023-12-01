import pytest
import torch
import torch.nn as nn
from typing import Any

from gyomei_trainer.builder.state import State
from gyomei_trainer.model.model import Model


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class FakeNN1:
    def __init__(self):
        self.a = 5

    def parameters(self):
        return (torch.Tensor() for _ in range(self.a))

    def forward(self):
        pass


class FakeNN2:
    def __init__(self):
        self.a = 5

    def parameters(self):
        return (torch.Tensor() for _ in range(self.a))

    def _get_name(self):
        pass


class TestModel:
    @pytest.mark.parametrize(
        "model, created", [(NeuralNetwork, True), (FakeNN1, False), (FakeNN2, False)]
    )
    def test_initialisation(self, model: Any, created: bool):
        """Test model with different model types.

        Args:
            model (Any): Tested model.
            created (bool): Should the scheduler be created.
        """
        raw_model = model()
        optimizer = torch.optim.Adam(params=raw_model.parameters())
        loss = torch.nn.MSELoss()
        device = "cpu"
        state = State()

        try:
            model = Model(raw_model, optimizer, loss, device)
            model.epoch_complete(state)
            assert created
        except AssertionError:
            assert not created
