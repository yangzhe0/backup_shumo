import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class BearingDataset(Dataset):
    def __init__(self, manifest_path: str, fixed_width: int):
        self.manifest = pd.read_csv(manifest_path)
        self.fixed_width = fixed_width
        self.classes = sorted(self.manifest['label'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.manifest)

    def _pad_or_truncate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape
        if width > self.fixed_width:
            return image[:, :self.fixed_width]
        elif width < self.fixed_width:
            padding = self.fixed_width - width
            return np.pad(image, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        return image

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        image = np.load(row['image_path'])
        image = self._pad_or_truncate(image)
        label_idx = self.class_to_idx[row['label']]
        return torch.from_numpy(image).unsqueeze(0).float(), torch.tensor(label_idx, dtype=torch.long)

class TargetDataset(Dataset):
    def __init__(self, manifest_path: str, fixed_width: int):
        self.manifest = pd.read_csv(manifest_path)
        self.fixed_width = fixed_width

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        image_path = self.manifest.iloc[idx]['image_path']
        image = np.load(image_path)
        height, width = image.shape
        if width > self.fixed_width:
            image = image[:, :self.fixed_width]
        elif width < self.fixed_width:
            padding_needed = self.fixed_width - width
            image = np.pad(image, ((0, 0), (0, padding_needed)), mode='constant', constant_values=0)
        return torch.from_numpy(image).unsqueeze(0).float()
