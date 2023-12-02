"""Implementation of BaseBuilder class."""

import abc
from typing import Dict, List, Optional, Any, Callable

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from gyomei_trainer.model import Model
from gyomei_trainer.metrics import Metrics
from gyomei_trainer.modules import EarlyStopping, LogsFormatter, Scheduler, create_logger
from gyomei_trainer.state import State, AverageValueMeter
from gyomei_trainer.utils import set_seed, create_experiment_folder


class BaseBuilder:
    """Base builder class to train networks."""

    def __init__(
        self,
        model: Model,
        train_loader: Optional[DataLoader],
        valid_loader: DataLoader,
        num_epoch: Optional[int],
        metrics: Dict[str, Any],
        main_metrics: Optional[List[str]],
        scheduler: lr_scheduler._LRScheduler,
        early_stopping_patience: Optional[int] = 5,
        project_path: Optional[str] = "./",
        seed: int = 777,
    ):
        """Base builder constructor.

        Allows you to train/validate/make prediction. Need to be careful
        because model should be prepared by gyomei-training model.

        Args:
            model (Model): Gyomei-like model.
            train_loader (Optional[DataLoader]): Train dataloader. In
                single validation run case may be set to None.
            valid_loader (DataLoader): Valid loader. Used for validation or
                test or prediction. In prediction case loader should return
                only data without target.
            num_epoch (Optional[int]): How many epoch should train. In
                single validation run case may be set to None.
            metrics (Dict[str, Any]): Metrics that should be calculated
                during run.
            main_metrics (Optional[List[str]]): Model save best accuracy
                by this metrics. In single validation run case may be
                set to None.
            scheduler (lr_scheduler._LRScheduler): Learning rate scheduler.
                To turf off set it to None.
            early_stopping_patience (Optional[int]): Stop training after
                not decreasing loss during patience epochs. To turf off
                set it to None. Defaults to 5.
            project_path (Optional[str]): Path to the project where the
                configs should be located. If the code is not run from the
                root folder, then you need to specify the directory path.
                To turn off saving models/configs set to None.
                Defaults to ./ .
            seed (int): Random seed allows to make a reproducible code.
                Defaults to 777.
        """
        set_seed(seed)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.num_epoch = num_epoch

        self.state = State()

        folder_path: Optional[str] = None
        if project_path is not None:
            folder_path = create_experiment_folder(project_path)

        self.state.update(folder_path=folder_path, main_metrics=main_metrics)
        self.logger = create_logger(self.state)
        self.model = model
        self.metrics = Metrics(metrics, device=model.device)
        self.scheduler = Scheduler(scheduler)
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        self._update_state()

        self.logs_formatter = LogsFormatter(self.state)

        self.logger.info("Builder has created")

    @abc.abstractmethod
    def train_epoch(self) -> None:
        """Train an expoch."""
        pass

    @abc.abstractmethod
    def valid_epoch(self) -> None:
        """Validate an epoch."""
        pass

    @abc.abstractmethod
    def predict(self, postprocessing: Callable) -> None:
        """Precidt on the data.

        Args:
            postprocessing (Callable): Postprocessing method.
        """
        pass

    def _update_state(self) -> None:
        """Update State after modules initialization."""
        metrics = {"": AverageValueMeter()}
        if self.metrics.metrics is not None:
            metrics = dict()
            for key in self.metrics.metrics.keys():
                metrics[key] = AverageValueMeter()

        self.state.update(
            logger=self.logger,
            device=self.model.device,
            metrics_name=metrics,
            loss_name=self.model.loss.loss_name,
        )
        self.logger.info("Updated State parameters")

    def _epoch_complete(self) -> None:
        """Run all methods that called after train and valid epoch."""
        for module in [
            self.model,
            self.scheduler,
            self.early_stopping,
            self.logs_formatter,
        ]:
            getattr(module, "epoch_complete")(state=self.state)
        self.state.epoch_complete()

    def fit(self) -> None:
        """Run model training."""
        while self.state.epoch < self.num_epoch and not self.early_stopping.stop:
            self.logger.info(f"Start epoch: {self.state.epoch}")
            self.state.start_timer()
            self.train_epoch()
            self.valid_epoch()
            self._epoch_complete()
            self.state.epoch += 1
