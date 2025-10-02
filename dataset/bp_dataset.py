"""
bp_dataset.py — PyTorch Dataset and DataLoader for BP Estimation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BP_Dataset(Dataset):
    """
    Dataset for Blood Pressure Estimation.

    Supports both:
      - PPG-only data   (X shape: [N, L])
      - PPG+ECG data    (X shape: [N, 2, L])
    """

    def __init__(self, npz_files, dataset_type="ppg_only", transform=None):
        """
        Args:
            npz_files (list[str]): list of .npz file paths
            dataset_type (str): "ppg_only" or "ppg_ecg"
            transform (callable, optional): optional transform
        """
        self.dataset_type = dataset_type
        self.transform = transform

        # Load and concatenate data
        X_list, y_list = [], []
        for file in npz_files:
            data = np.load(file)
            X_list.append(data["X"])
            y_list.append(data["y"])

        self.X = np.concatenate(X_list, axis=0)
        self.y = np.concatenate(y_list, axis=0)

        # Add channel dimension for PPG-only data
        if dataset_type == "ppg_only" and self.X.ndim == 2:
            self.X = np.expand_dims(self.X, axis=1)  # [N, 1, L]

        print(f"✅ Loaded {len(self.X)} samples from {len(npz_files)} file(s)")
        print(f"Input shape: {self.X.shape}, Labels shape: {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


def get_dataloader(npz_files, dataset_type="ppg_only", batch_size=64, shuffle=True):
    """
    Create a DataLoader from processed .npz dataset(s).

    Args:
        npz_files (list[str]): list of .npz file paths
        dataset_type (str): "ppg_only" or "ppg_ecg"
        batch_size (int): number of samples per batch
        shuffle (bool): whether to shuffle data each epoch

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = BP_Dataset(npz_files, dataset_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
