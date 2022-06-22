from typing import Optional
from os.path import join
import numpy as np
import random
import yaml
import os

import torch


def set_seed(seed: int):
    """Set seed to make reproducible process.

    Args:
        seed (int): The random seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_configs(folder_path: str, config_path: str):
    """Save configs to this experiment.

    Args:
        folder_path (str): Path to the experiment folder.
        config_path (str): Path to the config folder.
    """
    os.system(f'cp {config_path}/train.yaml '
              f'{folder_path}/train.yaml')
    os.system(f'cp {config_path}/data/dataset.yaml '
              f'{folder_path}/dataset.yaml')
    os.system(f'cp {config_path}/model/model.yaml '
              f'{folder_path}/model.yaml')


def create_experiment_folder(
        project_path: Optional[str]) -> Optional[str]:
    """Create experiment directory for model training.
    Doesn't create folder in case project_path is None.

    Args:
        project_path (str): Path to the project with configs.

    Returns:
        Optional[str]: Path to the directory from ./bin/train.py.
    """
    if project_path is None:
        return None

    assert os.path.exists(join(project_path, 'configs')), \
        "There is no config directory! Check project_path argument " \
        "in the Builder initialization"

    main_folder = join(project_path, 'experiments')
    os.makedirs(main_folder, exist_ok=True)

    config_path = join(project_path, 'configs')
    base_name = 'run'

    files_list = os.listdir(main_folder)
    folder_idx = sum([base_name in file for file in files_list])
    full_base_name = base_name + '_' + str(folder_idx)
    full_path = os.path.join(main_folder, full_base_name)

    os.makedirs(full_path)
    os.makedirs(os.path.join(full_path, 'models'))
    save_configs(folder_path=full_path, config_path=config_path)

    return full_path
