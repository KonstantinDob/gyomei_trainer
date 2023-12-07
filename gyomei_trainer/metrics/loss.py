"""Implementation of Gyomei Loss class."""

import numpy as np
from typing import Optional

import torch
from torch.nn.modules.loss import _Loss


class Loss(_Loss):
    """Create loss and prepare to Build."""

    def __init__(self, loss: Optional[torch.nn.Module]):
        """Loss constructor.

        Args:
            loss (torch.nn.Module, optional): Pytorch-like loss. Should
                have _get_name() method.

        Raises:
            TypeError: Incorrect loss type.
            AttributeError: Loss class doesn't have a method.
        """
        super(Loss, self).__init__()

        self.loss = loss
        self.loss_name: Optional[str] = None
        if loss is not None:
            if not isinstance(loss, torch.nn.Module):
                raise TypeError("Loss should be PyTorch-like")
            if not hasattr(loss, "_get_name"):
                raise AttributeError("Loss should have a _get_name() method!")
            self.loss_name = self.loss._get_name()

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Call loss.

        Args:
            *args: Positional args for parent class.
            **kwargs: Keyword args for parent class.

        Returns:
            torch.Tensor: Loss value.
        """
        loss_value = self.loss(*args, **kwargs)
        return loss_value

    def make_loss_value(self, loss: torch.Tensor) -> np.ndarray:
        """Prepare torch-like loss value to detached numpy.

        Args:
            loss (torch.Tensor): Tensor-like loss value.

        Returns:
            np.ndarray: Numpy-like loss value.
        """
        loss_value = loss.cpu().detach().numpy()
        return loss_value
