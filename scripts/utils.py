import json
import os
from src.dataset import MedicalDataset
from torch.utils.data import DataLoader


def create_dataloader(config, name, mode):
    dataset = MedicalDataset(
        name=name, 
        mode=mode, 
        root_dir=config.root[name],
        csv_file=f'b{mode}.csv'
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True, 
        num_workers=config.num_workers
    )
    return dataloader


def get_checkpoint(checkpoints_folder):
    paths = os.listdir(checkpoints_folder)
    for path in paths:
        if '.pth' in path:
            checkpoint = f'{checkpoints_folder}/{path}'
            return checkpoint
    return ''


def save_results(checkpoints_folder, ood_metrics):
    path = os.path.join(checkpoints_folder, 'results.json')
    with open(path, 'w') as f:
        json.dump(ood_metrics, f, indent=4)