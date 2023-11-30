"""Gyomei early stopping implementation."""

from typing import Optional
from gyomei_trainer.builder.state import State


class EarlyStopping:
    """Create early stopping method."""

    def __init__(self, patience: Optional[int]):
        """Early stopping constructor.

        Args:
            patience (int): How many epoch loss shouldn't decrease to stop
                training.
        """
        self.patience: Optional[int] = None
        if patience is not None:
            self.patience = None if patience < 1 else patience
        self.counter: int = 0
        self.stop: bool = False
        self.best_loss: Optional[float] = None

    def epoch_complete(self, state: State):
        """Check early stopping.

        If loss doesn't decrease during patient epoch then
        stop set to Ture that stop training.

        Args:
            state (State): State with main parameters.
        """
        if self.patience is None:
            return
        if self.best_loss is None:
            self.best_loss = state.loss_value_valid.value()[0]
        elif self.best_loss <= state.loss_value_valid.value()[0]:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                state.logger.info(
                    f"Early stop training. Loss doesn't change during {self.patience} epochs"
                )
        else:
            self.counter = 0
