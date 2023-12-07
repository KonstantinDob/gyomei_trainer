"""State calss implementation."""

import logging
import numpy as np
from time import time
from typing import Dict, List, Any, Union, Optional


class State:
    """Builder state."""

    def __init__(self):
        """State cinstructor.

        Allow you to pass arguments to various gyomei-trainig modules.
        You should be careful to get values from State because parameters
        may be None. For example, for only validation run an optimizer is
        not required.
        """
        self.iteration: int = 0
        self.epoch: Optional[int] = 0
        self.timer: Optional[float] = 0.0
        self.logger: Optional[logging.Logger] = None
        self.device: Optional[str] = None
        self.folder_path: Optional[str] = None
        self.metrics_name: Optional[Dict[str, AverageValueMeter]] = None
        self.main_metrics: Optional[List[str]] = None
        self.loss_name: Optional[str] = None
        self.loss_value_train: AverageValueMeter = AverageValueMeter()
        self.loss_value_valid: AverageValueMeter = AverageValueMeter()

    def start_timer(self) -> None:
        """Start timer."""
        self.timer = time()

    def update(self, **kwargs) -> None:
        """Update parameters in State.

        May have few parameters. Should be careful when update dict
        data.

        Args:
            **kwargs: Keyword args for parent class.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def epoch_complete(self) -> None:
        """Reset all epoch values to initial."""
        for key, value in self.metrics_name.items():
            value.reset()
        self.timer = 0
        self.loss_value_train.reset()
        self.loss_value_valid.reset()


class Meter(object):
    """Meter class implementation.

    Track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self) -> None:
        """Resets the meter to default settings."""
        pass

    def add(self, value: Any) -> None:
        """Log a new value to the meter.

        Args:
            value (Any): Next result to include.
        """
        pass

    def value(self) -> None:
        """Get the value of the meter in the current state."""
        pass


class AverageValueMeter(Meter):
    """Contain mean and std for parameter."""

    def __init__(self):
        """Average value meter Constructor.

        Allows you to easily contain data in parameter to analyze
        statistics, plow data e.t.c.
        """
        super(AverageValueMeter, self).__init__()
        self.n: int = 0
        self.sum: float = 0.0
        self.var: float = 0.0
        self.val: float = 0.0
        self.mean: Optional[Union[float, np.nan]] = np.nan
        self.mean_old: float = 0.0
        self.m_s: float = 0.0
        self.std: Optional[Union[float, np.nan]] = np.nan

        self.reset()

    def add(self, value: np.ndarray, n: int = 1) -> None:
        """Add value.

        Args:
            value (np.ndarray): Calculate mean and std for the value.
            n (int): Parameter to calculate mean and std. Defaults to 1.
        """
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = (value - n * self.mean_old) / float(self.n) + self.mean_old
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self) -> List[float]:
        """Return mean and std of parameter.

        Returns:
            list of float: Contain mean and std of parameter.
        """
        return self.mean, self.std

    def reset(self) -> None:
        """Reset all data."""
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
