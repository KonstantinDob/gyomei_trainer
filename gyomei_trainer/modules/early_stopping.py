from typing import Optional
from gyomei_trainer.builder.state import State


class EarlyStopping:
    """Create early stopping method.

    Args:
        patience (int): How many epoch loss shouldn't decrease to stop
            training.
    """
    def __init__(self, patience: int):
        self.patience = patience
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
        elif self.best_loss < state.loss_value_valid.value()[0]:
            self.counter += 1
            if self.counter > self.patience:
                self.stop = True
                state.logger.info("Early stop training. Loss doesn't "
                                  f"change during {self.patience} "
                                  f"epochs")
        else:
            self.counter = 0
