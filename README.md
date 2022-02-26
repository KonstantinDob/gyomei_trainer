# Gyomei trainer

Gyomei is a package for training pytorch-like neural networks.


## Installation

Requirements:
* Python == 3.8
* numpy==1.22.1
* PyYAML==6.0
* tensorboard==2.8.0
* torch==1.10.2

From pip:
```
python -m build
pip install dist/gyomei_trainer-1.0.2-py3-none-any.whl 
```

From source:
```
pip install -U git+https://github.com/KonstantinDob/gyomei_trainer.git
```

## Docker 

To use docker with GPU you need *nvidia-docker == 2.9.0*.

Build project:

```
make build
```

Run in interactive mode:

```
make run
```

## Project structure

Before starting training, an individual directory is created in which 
the current configs, Neural Network weights, logs, and data for 
running tensorboard are saved.

To fit model you need to create a specific config architecture.
The root of the project should contain the configs directory. The 
package copy 3 configs that shall be in the certain paths.

    .
    ├── ...
    ├── configs 
    │   ├── data                
    │   │   └── dataset.yaml   
    │   ├── model               
    │   │   └── model.yaml
    │   └── train.yaml          
    └── ...

* `dataset.yaml` - Contain path to dataset and augmentation params.
* `model.yaml` - Contain info about backbone, base model, number of 
channels etc.
* `train.yaml` - Contain Loss/Scheduler/Optimizer info, number of epoch 
and batch size.

## Example

Base fit example with empty Dataset using Segmentation Models Pytorch.

```
import torch
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


def main():
    """Dataloader you may find in gyomei-detection project."""
    train_dataloader = DataLoader(DummyDataset())
    valid_dataloader = DataLoader(DummyDataset())

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
    
    # To create folder with configs and saved weights need to
    # set project_path to real project path with configs.
    trainer = Builder(model=model, train_loader=train_dataloader,
                      valid_loader=valid_dataloader, num_epoch=20,
                      metrics=metrics, main_metrics=main_metric,
                      scheduler=scheduler, early_stopping_patience=5,
                      project_path=None, seed=666)

    trainer.fit()


if __name__ == "__main__":
    main()
```

