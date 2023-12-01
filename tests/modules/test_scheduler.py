import pytest
import torch
import torch.nn as nn

from gyomei_trainer.modules.lr_scheduler import Scheduler


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


class TestScheduler:
    @pytest.mark.parametrize(
        "scheduler_name, created",
        [("linear", True), ("None", False), ("lambda", True), ("Lol", False)],
    )
    def test_scheduler(self, scheduler_name: str, created: bool):
        """Test learning rate scheduler.
        AssertError activates in case of None scheduler and
        AttributeError activates then step() method is used from
        None type.

        Args:
            scheduler_name (str): Name of the scheduler. There is a
            linear and lambda one now.
            created (bool): Should the scheduler be created.

        Returns:

        """
        model = NeuralNetwork()
        optimizer = torch.optim.Adam(params=model.parameters())

        scheduler = None
        if scheduler_name == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        elif scheduler_name == "lambda":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=[lambda epoch: epoch // 30]
            )
        try:
            lr_scheduler = Scheduler(scheduler)
            optimizer.step()
            lr_scheduler.scheduler.step()
            assert created
        except (AssertionError, AttributeError):
            assert not created
