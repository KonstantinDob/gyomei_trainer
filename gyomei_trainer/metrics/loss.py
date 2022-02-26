import numpy as np
from typing import Optional

import torch
from torch.nn.modules.loss import _Loss


class Loss(_Loss):
    """Create loss and prepare to Build.

    Args:
        loss (Optional[torch.nn.Module]): Pytorch-like loss. Should
            have _get_name() method.
    """
    def __init__(self, loss: Optional[torch.nn.Module]):
        super(Loss, self).__init__()

        self.loss = loss
        self.loss_name: Optional[str] = None
        if loss is not None:
            assert isinstance(loss, torch.nn.Module), \
                "Loss should be PyTorch-like"
            assert hasattr(loss, '_get_name'), \
                "Loss should have a _get_name() method!"
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
