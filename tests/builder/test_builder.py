import pytest
import torch
from typing import Callable
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import losses

from gyomei_trainer.model.model import Model
from gyomei_trainer.builder.builder import Builder


class DummyDataset(Dataset):
    """Create empty Dataset to test Builder."""
    def __len__(self):
        return 0

    def __getitem__(self, item):
        return [], []


@pytest.fixture
def create_builder() -> Builder:
    """Create base Builder with empty dataloader.
    Also, this builder doesn't create a folders.

    Returns:
        Builder: Base builder.
    """
    train_dataloader = DataLoader(DummyDataset())
    valid_dataloader = DataLoader(DummyDataset())

    smp_model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=2
    )
    optimizer = torch.optim.Adam(params=smp_model.parameters())
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    metrics = dict()
    metrics['fscore'] = smp.utils.metrics.Fscore(threshold=0.5)
    metrics['iou'] = smp.utils.metrics.IoU(threshold=0.5)
    main_metric = ['iou']

    loss = losses.JaccardLoss(mode='multilabel', smooth=0)

    model = Model(model=smp_model, optimizer=optimizer,
                  loss=loss, device='cpu')

    trainer = Builder(model=model, train_loader=train_dataloader,
                      valid_loader=valid_dataloader, num_epoch=20,
                      metrics=metrics, main_metrics=main_metric,
                      scheduler=scheduler, early_stopping_patience=5,
                      project_path=None, seed=666)
    return trainer


class TestBuilder:

    def test_state_update(self, create_builder: Callable):
        """Test update of state."""
        trainer = create_builder

        assert trainer.state.device == 'cpu'
        assert trainer.state.loss_name == 'JaccardLoss'
        assert trainer.state.main_metrics == ['iou']
        assert trainer.state.epoch == 0
        assert trainer.state.iteration == 0

        trainer.state.update(epoch=10, iteration=100)
        assert trainer.state.epoch == 10
        assert trainer.state.iteration == 100

        trainer.state.update(loss_name='DiceLoss',
                             main_metrics=['recall', 'f1'])
        assert trainer.state.loss_name == 'DiceLoss'
        assert trainer.state.main_metrics == ['recall', 'f1']

    def test_fit(self, create_builder: Callable):
        """Test fit with empty dataset."""
        trainer = create_builder
        trainer.fit()
        assert trainer.state.epoch == 20
