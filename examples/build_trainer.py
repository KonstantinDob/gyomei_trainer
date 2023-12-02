"""Example of building the trainer."""

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import losses

from gyomei_trainer.model import Model
from gyomei_trainer.builder import Builder


class DummyDataset(Dataset):
    """Create empty Dataset to test Builder."""

    def __len__(self) -> int:
        """Return 0 as dataset length.

        Returns:
            int: Length of dummy dataset.
        """
        return 0

    def __getitem__(self, item: Any) -> list:
        """Return empty lists for dummy dataset.

        Args:
            item (Any): Any type of data.

        Returns:
            list: Empty lists.
        """
        return [], []


def main():
    """Dataloader you may find in gyomei-detection project."""
    train_dataloader = DataLoader(DummyDataset())
    valid_dataloader = DataLoader(DummyDataset())

    smp_model = smp.Unet(
        encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=2
    )
    optimizer = torch.optim.Adam(
        params=smp_model.parameters(), lr=0.001, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.999)

    metrics = dict()
    metrics["fscore"] = smp.utils.metrics.Fscore(threshold=0.5)
    metrics["iou"] = smp.utils.metrics.IoU(threshold=0.5)

    main_metric = ["iou"]

    loss = losses.JaccardLoss(mode="multilabel", smooth=0)

    model = Model(model=smp_model, optimizer=optimizer, loss=loss, device="cuda")

    # To create folder with configs and saved weights need to
    # set project_path to real project path with configs.
    trainer = Builder(
        model=model,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        num_epoch=20,
        metrics=metrics,
        main_metrics=main_metric,
        scheduler=scheduler,
        early_stopping_patience=5,
        project_path=None,
        seed=666,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
