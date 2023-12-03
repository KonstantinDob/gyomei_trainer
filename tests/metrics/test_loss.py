import pytest
import torch
from typing import Any

from gyomei_trainer.metrics import Loss


class TestLoss:
    @pytest.mark.parametrize(
        "raw_loss, created",
        [(torch.nn.MSELoss(), True), (torch.nn.BCELoss(), True), ([], False)],
    )
    def test_loss_init(self, raw_loss: Any, created: bool):
        """Test lsos initialization.

        Args:
            raw_loss (Any): Any loss function.
            created (bool): Should the scheduler be created.
        """
        try:
            loss_value = torch.tensor([1, 2, 3])
            loss = Loss(raw_loss)
            loss.make_loss_value(loss_value)
            assert created
        except TypeError:
            assert not created
