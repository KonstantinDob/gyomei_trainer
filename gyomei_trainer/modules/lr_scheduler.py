"""Gyomei scheduler implementation."""

from typing import Any
from torch.optim import lr_scheduler


class Scheduler:
    """Init scheduler class."""

    def __init__(self, scheduler: lr_scheduler._LRScheduler):
        """Scheduler constructor.

        May be various types, but it's necessary the scheduler has the
        step() method and by PyTorch-like.

        Args:
            scheduler (lr_scheduler._LRScheduler): Input scheduler. Should be PyTorch-like.

        Raises:
            AttributeError: Scheduler class doesn't have a method.
        """
        if scheduler is not None:
            if not hasattr(scheduler, "step"):
                raise AttributeError("Scheduler should have step() method")
        self.scheduler = scheduler

    def epoch_complete(self, state: Any) -> None:
        """Make scheduler step after end of training epoch.

        Args:
              state (Any): State with main parameters.
        """
        if self.scheduler is not None:
            state.logger.info("Make scheduler step")
            self.scheduler.step()
