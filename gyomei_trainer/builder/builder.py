"""Implementation of Builder class."""

from typing import Callable

from gyomei_trainer.builder.base import BaseBuilder


class Builder(BaseBuilder):
    """Builder class to run training."""

    def __init__(self, *args, **kwargs):
        """Builder constructor.

        Allows you to train/validate/make prediction. Need to be careful
        because model should be prepared by gyomei-training model.

        Args:
            *args: Positional args for parent class.
            **kwargs: Keyword args for parent class.

        Examples:
            Create Builder. Detailed Model initialization showed in the
            model class example.

            model = Model(...)

            metrics = metrics = dict()
            metrics['fscore'] = smp.utils.metrics.Fscore(threshold=0.5)
            metrics['iou'] = smp.utils.metrics.IoU(threshold=0.5)
            metrics['recall'] = smp.utils.metrics.Recall(threshold=0.5)

            main_metrics = ['fscore', 'iou']

            scheduler = Scheduler(...)

            train_loader, valid_loader = create_loaders()

            trainer = Builder(
                model=model,
                train_loader=train_dataloader,
                valid_loader=valid_dataloader,
                num_epoch=20,
                metrics=metrics,
                main_metrics=main_metric,
                scheduler=scheduler,
                early_stopping_patience=5,
                seed=777
            )

            trainer.fit()
        """
        super().__init__(*args, **kwargs)

    def train_epoch(self) -> None:
        """Train one epoch."""
        for batch in self.train_loader:
            result = self.model.train_step(batch)
            loss = self.model.loss.make_loss_value(result["loss"])
            self.state.loss_value_train.add(value=loss)
            self.state.iteration += 1

    def valid_epoch(self) -> None:
        """Make validation for one epoch.

        Can be used to validation without training. In this case should
        set project_path to None.

        Examples:
            1. Builder can be crated as in Builder example.

            build = Builder(...)
            build.validate()

            2. Most of params may be None.

            build = Builder(
                model=model,
                train_loader=None,
                valid_loader=valid_dataloader,
                num_epoch=None,
                metrics=metrics,
                main_metrics=None,
                scheduler=None,
                early_stopping_patience=None,
                project_path=None,
                seed=777
            )
            build.valid_epoch()
        """
        for batch in self.valid_loader:
            result = self.model.valid_step(batch)
            loss = self.model.loss.make_loss_value(result["loss"])
            metrics = self.metrics.calculate_metrics(
                result["prediction"], result["target"]
            )
            self.state.loss_value_valid.add(value=loss)
            for key, value in metrics.items():
                self.state.metrics_name[key].add(value=value)
        if self.state.folder_path is None:
            self.logs_formatter.epoch_complete(self.state)

    def predict(self, postprocessing: Callable) -> None:
        """Make predictions.

        Args:
            postprocessing (Callable): Function allows you to operate
                with processed data. For example, it may save images or text.
        """
        for batch in self.valid_loader:
            result = self.model.predict(batch)
            postprocessing(result)
