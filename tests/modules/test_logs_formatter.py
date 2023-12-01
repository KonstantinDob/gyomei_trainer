import pytest
import logging
from typing import Dict

from gyomei_trainer.builder.state import State, AverageValueMeter
from gyomei_trainer.modules.logs_formatter import LogsFormatter, make_epoch_log


class TestFormatter:
    @pytest.mark.parametrize(
        "timer, loss_name, loss_value_valid, metrics_name, epoch",
        [
            (-5.0, "", AverageValueMeter(), {"": AverageValueMeter()}, -20),
            (0.0, "123", AverageValueMeter(), {}, 0),
            (
                1e3,
                "-" * int(1e2),
                AverageValueMeter(),
                {"0" * int(1e2): AverageValueMeter()},
                1e6,
            ),
        ],
    )
    def test_logs_formatter(
        self,
        timer: float,
        loss_name: str,
        loss_value_valid: AverageValueMeter,
        metrics_name: Dict[str, AverageValueMeter],
        epoch: int,
    ):
        """Test creating and updating LogFormatter.

        Args:
            timer (float): Time of train begging.
            loss_name (str): Loss name.
            loss_value_valid (AverageValueMeter): Structure contained
                mean and std of loss values.
            metrics_name (Dict[str, AverageValueMeter]): Structure
                contained mean and std of every metric.
            epoch (int): Number of current epoch.
        """
        logging.basicConfig(format="%(message)s", level=logging.INFO)
        logger = logging.getLogger("gyomei_detection")

        state = State()
        state.update(
            logger=logger,
            timer=timer,
            loss_name=loss_name,
            loss_value_valid=loss_value_valid,
            metrics_name=metrics_name,
            epoch=epoch,
        )
        formatter = LogsFormatter(state)
        formatter.epoch_complete(state)

    @pytest.mark.parametrize(
        "seconds, metric_data, epoch",
        [
            (10.0, {"loss": AverageValueMeter(), "accuracy": AverageValueMeter()}, 5),
            (-5.2, {"": AverageValueMeter(), "--": AverageValueMeter()}, -5),
            (66.6, {}, 0),
        ],
    )
    def test_make_epoch_log(
        self, seconds: float, metric_data: Dict[str, AverageValueMeter], epoch: int
    ):
        """Test that logs created properly.

        Args:
            seconds (float): Time required at 1 epoch.
            metric_data (Dict[str, AverageValueMeter]): Dict with
                metrics parameters.
            epoch (int): Number of current epoch.
        """
        out = make_epoch_log(seconds, metric_data, epoch)
        assert isinstance(out, str)
