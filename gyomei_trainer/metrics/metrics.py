"""Implementation of Gyomei metrics."""

from typing import Dict, Any


class Metrics:
    """Prepare metrics to Build format."""

    def __init__(self, metrics: Dict[str, Any], device: str):
        """Metrics constructor.

        Args:
            metrics (dict of str: any): Dict with metrics. Keys are metric
                names and values are metric classes.
            device (str): On that device metrics should be loaded.

        Examples:
            Create with segmentation models pytorch project.

            metrics = dict()
            metrics['fscore'] = smp.utils.metrics.Fscore(threshold=0.5)
            metrics['iou'] = smp.utils.metrics.IoU(threshold=0.5)
        """
        self.metrics = metrics
        self.device = device
        self._to_device()

    def _to_device(self) -> None:
        """Load model and loss to the device."""
        if self.metrics is not None:
            for key, value in self.metrics.items():
                self.metrics[key] = value.to(self.device)

    def calculate_metrics(self, prediction: Any, target: Any) -> Dict[str, Any]:
        """Calculate metrics based on prediction and target.

        Args:
            prediction (Any): Model prediction.
            target (Any): Target to that the model trains.

        Returns:
            dict of str: any: Dict with metrics values. Keys are
                metrics names and values are metric values.
        """
        metric_values = dict.fromkeys(self.metrics.keys(), 0)
        for key, value in self.metrics.items():
            metric = self.metrics[key]
            metric_value = metric(prediction, target)
            metric_values[key] = metric_value.cpu().detach().numpy()
        return metric_values
