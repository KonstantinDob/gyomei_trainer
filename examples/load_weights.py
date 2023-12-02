"""Implementation of weights loading."""

import torch

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import losses

from gyomei_trainer.model import Model


def main():
    """Example of weight laoding."""
    smp_model = smp.Unet(
        encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=2
    )
    optimizer = torch.optim.Adam(
        params=smp_model.parameters(), lr=0.001, betas=(0.9, 0.999)
    )

    metrics = dict()
    metrics["fscore"] = smp.utils.metrics.Fscore(threshold=0.5)
    metrics["iou"] = smp.utils.metrics.IoU(threshold=0.5)

    loss = losses.JaccardLoss(mode="multilabel", smooth=0)

    model = Model(model=smp_model, optimizer=optimizer, loss=loss, device="cuda")
    model.load_model("path to model.pth")


if __name__ == "__main__":
    main()
