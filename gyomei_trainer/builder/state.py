import logging
import numpy as np
from time import time
from typing import Dict, List, Any, Optional


class State:
    """Builder state.
    Allow you to pass arguments to various gyomei-trainig modules.
    You should be careful to get values from State because parameters
    may be None. For example, for only validation run an optimizer is
    not required.
    """
    def __init__(self):
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

    def start_timer(self):
        """Start timer."""
        self.timer = time()

    def update(self, **kwargs):
        """Update parameters in State.
        May have few parameters. Should be careful when update dict
        data.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def epoch_complete(self):
        """Reset all epoch values to initial."""
        for key, value in self.metrics_name.items():
            value.reset()
        self.timer = 0
        self.loss_value_train.reset()
        self.loss_value_valid.reset()


class Meter(object):
    """Keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface
    for all meters to follow.
    """
    def reset(self):
        """Resets the meter to default settings."""
        pass

    def add(self, value: Any):
        """Log a new value to the meter.

        Args:
            value (Any): Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state.'"""
        pass


class AverageValueMeter(Meter):
    """Contain mean and std for parameter.
    Allows you to easily contain data in parameter to analyze
    statistics, plow data e.t.c.
    """
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.n: int = 0
        self.sum: float = 0.0
        self.var: float = 0.0
        self.val: float = 0.0
        self.mean: Optional[float, np.nan] = np.nan
        self.mean_old: float = 0.0
        self.m_s: float = 0.0
        self.std: Optional[float, np.nan] = np.nan

        self.reset()

    def add(self, value, n=1):
        """Add value."""
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
            self.mean = (value - n * self.mean_old) / \
                float(self.n) + self.mean_old
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        """Return mean and std of parameter.

        Returns:
            List[float]: Contain mean and std of parameter.
        """
        return self.mean, self.std

    def reset(self):
        """Reset all data."""
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
