import numpy as np
import os
import pandas as pd
from src.utils import open_file
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor


def get_label(name, df, index):
    if name == 'skeletal-age':
        label = df.iloc[index, 1] - 1  # there is no 0 month label
    elif name == 'mura':
        label = df.iloc[index, 1]
    elif name == 'retina':
        label = df.iloc[index, 1]
    elif name == 'mimic-cxr':
        label = df.iloc[index, 2]
    else:
        raise NotImplementedError
    return np.array(label, dtype=np.long)


class MedicalDataset(Dataset):
    def __init__(self, name, mode, root_dir, csv_file, im_size=224):
        super(MedicalDataset, self).__init__()
        self.name = name
        self.mode = mode
        assert self.mode in ['train', 'test']
        assert self.name in ['retina', 'skeletal-age', 'mura', 'mimic-cxr']
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(root_dir, csv_file))
        
        # Create the transformations
        if self.mode == 'test':
            self.input_transforms = Compose([
                Resize((im_size, im_size)),
                ToTensor()
            ])
        else:
            self.input_transforms = Compose([
                RandomResizedCrop((im_size, im_size)),
                RandomHorizontalFlip(),
                Resize((im_size, im_size)),
                ToTensor()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_arr = open_file(f'{self.root_dir}/{self.df.iloc[index, 0]}')
        inputs = self.input_transforms(image_arr)
        label = get_label(self.name, self.df, index)
        return inputs, label
