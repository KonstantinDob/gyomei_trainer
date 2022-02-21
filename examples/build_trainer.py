import torch
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import losses

from gyomei_trainer.model.model import Model
from gyomei_trainer.builder.builder import Builder


def main():
    """Dataloader you may find in gyomei-detection project."""
    train_dataloader = DataLoader(Dataset)
    valid_dataloader = DataLoader(Dataset)

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
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.999)

    metrics = dict()
    metrics['fscore'] = smp.utils.metrics.Fscore(threshold=0.5)
    metrics['iou'] = smp.utils.metrics.IoU(threshold=0.5)

    main_metric = ['iou']

    loss = losses.JaccardLoss(mode='multilabel', smooth=0)

    model = Model(model=smp_model, optimizer=optimizer,
                  loss=loss, device='cuda')

    trainer = Builder(model=model, train_loader=train_dataloader,
                      valid_loader=valid_dataloader, num_epoch=20,
                      metrics=metrics, main_metrics=main_metric,
                      scheduler=scheduler, early_stopping_patience=5,
                      seed=666)

    # To fit you need to use real dataloader.
    trainer.fit()


if __name__ == "__main__":
    main()
