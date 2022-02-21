import logging
from time import time
from os.path import join
from datetime import datetime
from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter

from gyomei_trainer.builder.state import State


def create_logger(state: State) -> logging.Logger:
    """Create logger.
    Save logs to file in training mode. To turn off training need to set
    up folder_path parameter to None.

    Args:
        state (State): State with main parameters.

    Returns:
        logging.Logger: General logger.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    logger = logging.getLogger('gyomei_detection')

    if state.folder_path is not None:
        fh = logging.FileHandler(join(state.folder_path, "std.log"))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return logger


def make_epoch_log(seconds: float,
                   metric_data: Dict[str, Any], epoch: int) -> str:
    """Create the log message basen on input parameters.

    Args:
        seconds (float): Time spent on train and valid epoch.
        metric_data (Dict[str, Any]): Dict with loss and metrics.
        epoch (int): Num of epoch.

    Returns:
        str: Created log message.
    """
    required_time = datetime.fromtimestamp(
        seconds).strftime("%M:%S")
    metrics_logs = ["{} - {:.4}".format(key, val.value()[0]) for
                    key, val in metric_data.items()]

    epoch_log = f"Epoch {epoch} finished. " \
                f"Required time: {required_time}. " \
                f"Valid metrics: {', '.join(metrics_logs)}"
    return epoch_log


class LogsFormatter:
    """Logs formatter also create tensorboard logs.

    Args:
        state (State): State with main parameters.
    """
    def __init__(self, state: State):
        self.writer = None
        if state.folder_path is not None:
            self.writer = SummaryWriter(state.folder_path)

    def epoch_complete(self, state: State):
        """Create and save logs after training and validation epoch."""
        seconds = time() - state.timer
        data = {state.loss_name: state.loss_value_valid}
        data.update(state.metrics_name)
        epoch_log = make_epoch_log(seconds, data, state.epoch)
        state.logger.info(epoch_log)

        if self.writer is not None:
            # Write data to tensorboard.
            self.writer.add_scalar(
                "Train loss",
                state.loss_value_train.value()[0],
                state.epoch
            )
            self.writer.add_scalar(
                "Valid loss",
                state.loss_value_valid.value()[0],
                state.epoch
            )
            for key, val in state.metrics_name.items():
                self.writer.add_scalar(
                    f"{key}", val.value()[0], state.epoch)
