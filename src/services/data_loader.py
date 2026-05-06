"""
PyTorch datasets and dataloaders.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Any
import random

class MLPDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray]):
        noisy = data["noisy_samples"]
        labels = data["frequency_label"]
        self.X = torch.tensor(np.concatenate([noisy, labels], axis=1), dtype=torch.float32)
        self.y = torch.tensor(data["target_output"], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray]):
        noisy = data["noisy_samples"]  # (N, 10)
        labels = data["frequency_label"]  # (N, 4)
        N, seq_len = noisy.shape
        _, num_classes = labels.shape
        
        # We need shape (N, 10, 5)
        # Repeat labels across seq_len
        labels_seq = np.repeat(labels[:, np.newaxis, :], seq_len, axis=1) # (N, 10, 4)
        noisy_seq = noisy[:, :, np.newaxis] # (N, 10, 1)
        
        self.X = torch.tensor(np.concatenate([noisy_seq, labels_seq], axis=2), dtype=torch.float32)
        self.y = torch.tensor(data["target_output"], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, seed: int = 42) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        worker_init_fn=seed_worker,
        generator=g
    )
