import os
import torch
from typing import Dict, Any

from gyomei_trainer.metrics.loss import Loss
from gyomei_trainer.builder.state import State


class Model:
    """Create model based on parameters.
    Model must be prepared by this class to be loaded to
    Builder. Input model can be various pytorch-like types but
    forward() and _get_name() methods should be present.

    Args:
        model (Any): Pytorch-like model. Should have forward() and
            _get_name() methods.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss (Any): Pytorch-like loss.
        device (str): On that device model should be loaded.

    Examples:

        Create with segmentation models pytorch project.

        smp_model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        )

        optimizer = torch.optim.Adam(
            params=smp_model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999)
        )

        loss = smp.losses.JaccardLoss(mode='multilabel', smooth=0)

        model = Model(model=smp_model, optimizer=optimizer,
                      loss=loss, device='cuda')
    """
    def __init__(self, model: Any, optimizer: torch.optim.Optimizer,
                 loss: Any, device: str):
        assert hasattr(model, 'forward'), \
            "Model should have a forward() method!"
        assert hasattr(model, '_get_name'), \
            "Model should have a _get_name() method!"

        self.model = model
        self.model_name = self.model._get_name()
        self.optimizer = optimizer
        self.loss = Loss(loss)
        self._device = device
        self.best_metrics_value: float = 0.0

        self._to_device()

    def _to_device(self):
        """Load model and loss to the device."""
        self.model.to(self._device)
        self.loss.to(self._device)

    @property
    def device(self) -> str:
        """Return current device."""
        return self._device

    @device.setter
    def device(self, device_name: str):
        """Change device type.

        Args:
            device_name (str): Which device should use.
        """
        self._device = device_name
        self._to_device()

    def epoch_complete(self, state: State):
        """Save model after training and validation epoch.

        Args:
            state (State): State with main parameters.
        """
        if state.folder_path is None:
            return

        self._save_model(state, 'last')
        state.logger.info('Latest model saved')

        metric_form = {}
        metrics_value = 0
        for metric_name in state.main_metrics:
            mean_value = state.metrics_name[metric_name].value()[0]
            metrics_value += mean_value
            metric_form.update(
                {metric_name: mean_value}
            )

        if metrics_value > self.best_metrics_value:
            self.best_metrics_value = metrics_value
            self._save_model(state, 'best')

            form_data = ''
            for key, value in metric_form.items():
                form_data += f'{key}: {value}'
            state.logger.info(f'Best model saved with '
                              f'following metric: {form_data}')

    def _save_model(self, state: State, mode: str):
        """Save pytorch model.

        Args:
            state (State): State with main parameters.
            mode (str): Add mode label to saved model name.
        """
        model_name = f'{mode}_{self.model_name}.pth'
        save_path = os.path.join(state.folder_path,
                                 'models', model_name)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
        }, save_path)
        state.logger.info("Model saving completed")

    def load_model(self, file_path: str):
        """Load gyomei-like Model.

        Args:
            file_path (str): Path to the weights.
        """
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']

    def train_step(self, batch) -> Dict[str, Any]:
        """Train model on a step.

        Args:
            batch (Any): Data on that one iteration of training
                will run. Contain train data and target.

        Returns:
            Dict[str, Any]: Result of one training step.
        """
        data, target = batch

        self.model.train()
        self.optimizer.zero_grad()
        data, target = data.to(self._device), target.to(self._device)
        prediction = self.model.forward(data)
        loss = self.loss(prediction, target)
        loss.backward()
        self.optimizer.step()
        return {'prediction': prediction,
                'target': target,
                'loss': loss}

    def valid_step(self, batch) -> Dict[str, Any]:
        """Validate model on a step.

        Args:
            batch (Any): Data on that one iteration of validation
                will run. Contain train data and target.

        Returns:
            Dict[str, Any]: Result of one validation step.
        """
        data, target = batch

        self.model.eval()
        with torch.no_grad():
            data, target = data.to(self.device), target.to(self.device)
            prediction = self.model.forward(data)
            loss = self.loss(prediction, target)
        return {'prediction': prediction,
                'target': target,
                'loss': loss}

    def predict(self, data) -> Any:
        """MAke prediction.

        Args:
            data (Any): Data on that prediction will run.

        Returns:
            Any: Return result of prediction
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            prediction = self.model.forward(data)
        return prediction
