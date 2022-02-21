import numpy as np
import torch
from typing import Any
from torch.nn.modules.loss import _Loss


class Loss(_Loss):
    """Create loss and prepare to Build.

    Args:
        loss (Any): Pytorch-like loss. Should have _get_name() method.
    """
    def __init__(self, loss: Any):
        assert hasattr(loss, '_get_name'), \
            "Loss should have a _get_name() method!"

        super(Loss, self).__init__()
        self.loss = loss
        self.loss_name = self.loss._get_name()

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Call loss."""
        loss_value = self.loss(*args, **kwargs)
        return loss_value

    def make_loss_value(self, loss: torch.Tensor) -> np.ndarray:
        """Prepare torch-like loss value to detached numpy.

        Args:
            loss (torch.Tensor): Tensor-like loss value.

        Return:
            np.ndarray: Numpy-like loss value.
        """
        loss_value = loss.cpu().detach().numpy()
        return loss_value
