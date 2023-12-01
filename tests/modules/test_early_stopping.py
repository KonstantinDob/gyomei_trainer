import pytest
import logging
from typing import Optional

from gyomei_trainer.builder.state import State
from gyomei_trainer.modules.early_stopping import EarlyStopping


class TestStopping:
    @pytest.mark.parametrize("patience", [1, 5, -2, 0, None])
    def test_stopping_process(self, patience: Optional[int]):
        """Test early stopping model with different patient values.

        Args:
            patience (Optional[int]): Patient of EarlyStopping.
        """
        early_stopping = EarlyStopping(patience)

        logging.basicConfig(format="%(message)s", level=logging.INFO)
        logger = logging.getLogger("gyomei_detection")
        state = State()
        state.logger = logger
        state.loss_value_valid.add(0)

        patience = 0 if patience is None else patience
        for epoch in range(patience + 1):
            assert not early_stopping.stop
            early_stopping.epoch_complete(state)
        if early_stopping.patience is not None:
            assert early_stopping.stop
