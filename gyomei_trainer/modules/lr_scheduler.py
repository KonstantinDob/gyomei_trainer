from typing import Any
from gyomei_trainer.builder.state import State


class Scheduler:
    """Init scheduler class.
    May be various types but it's necessary the scheduler has the
    step() method.

    Args:
        scheduler (Any): Input scheduler.
    """
    def __init__(self, scheduler: Any):
        if scheduler is not None:
            assert hasattr(scheduler, 'step'), \
                "Scheduler should have step() method"
        self.scheduler = scheduler

    def epoch_complete(self, state: State):
        """Make scheduler step after end of training epoch.

        Args:
              state (State): State with main parameters.
        """
        if self.scheduler is not None:
            state.logger.info('Make scheduler step')
            self.scheduler.step()
