from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ot
import pandas as pd

class RealBaseDataset(Dataset):
    def __init__(self, y, a):
        self.y = y
        self.a = a

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.y[idx], self.a[idx]

class ConditionedDataset(Dataset):
    def __init__(self, base_dataset, x_tensor, counter_y_tensor=None, mu0_tensor=None, mu1_tensor=None,
                 mode="Training"):
        assert len(base_dataset) == len(x_tensor)
        self.base_dataset = base_dataset
        self.x_tensor = x_tensor
        self.mode = mode
        if self.mode == "Test":
            self.counter_y_tensor = counter_y_tensor
            self.mu0_tensor = mu0_tensor
            self.mu1_tensor = mu1_tensor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        y, a = self.base_dataset[idx]
        x = self.x_tensor[idx]
        if self.mode == "Training":
            return torch.tensor([y]), a, x
        else:
            return torch.tensor([y]), a, x, self.counter_y_tensor[idx], self.mu0_tensor[idx], self.mu1_tensor[idx]

class CounterfactualDatasetBuilder:
    def __init__(self, df, device="cpu", split_ratio=0.8):
        self.df = df
        self.device = device
        self.split_ratio = split_ratio

    def build(self):
        a_vals = self.df.iloc[:, 0].astype(int).values
        y0_vals = self.df.iloc[:, 1].values
        y1_vals = self.df.iloc[:, 2].values
        mu0_vals = self.df.iloc[:, 3].values
        mu1_vals = self.df.iloc[:, 4].values

        factual_y = np.where(a_vals == 0, y0_vals, y1_vals)
        counter_y = np.where(a_vals == 0, y1_vals, y0_vals)
        x_vals = self.df.iloc[:, 5:].values

        x_tensor = torch.tensor(x_vals, dtype=torch.float32)
        y_tensor = torch.tensor(factual_y, dtype=torch.float32)
        a_tensor = torch.tensor(a_vals, dtype=torch.int64)
        counter_y_tensor = torch.tensor(counter_y, dtype=torch.float32)
        mu0_tensor = torch.tensor(mu0_vals, dtype=torch.float32)
        mu1_tensor = torch.tensor(mu1_vals, dtype=torch.float32)

        # Train/test split
        total_samples = y_tensor.shape[0]
        indices = torch.randperm(total_samples)
        split = int(self.split_ratio * total_samples)
        train_idx = indices[:split]
        test_idx = indices[split:]

        train_x, test_x = x_tensor[train_idx], x_tensor[test_idx]
        train_y, test_y = y_tensor[train_idx], y_tensor[test_idx]
        train_a, test_a = a_tensor[train_idx], a_tensor[test_idx]
        test_counter_y = counter_y_tensor[test_idx]
        test_mu0_tensor = mu0_tensor[test_idx]
        test_mu1_tensor = mu1_tensor[test_idx]

        # Wrap in datasets
        train_dataset = ConditionedDataset(
            RealBaseDataset(train_y, train_a), train_x)

        test_dataset = ConditionedDataset(
            RealBaseDataset(test_y, test_a), test_x,
            counter_y_tensor=test_counter_y,
            mu0_tensor=test_mu0_tensor,
            mu1_tensor=test_mu1_tensor,
            mode="Test"
        )

        return train_dataset, test_dataset



def wasserstein_dist(x: torch.Tensor, y: torch.Tensor, k: int = 1) -> float:
    """
    Compute empirical Wasserstein-k distance between two empirical distributions using optimal transport.
    Args:
        x (Tensor): shape (N,) or (N,D)
        y (Tensor): shape (M,) or (M,D)
        k (int): order of the distance (W_k), typically 1 or 2

    Returns:
        float: Wasserstein distance W_k(x, y)
    """
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    if x_np.ndim == 1:
        x_np = x_np[:, None]
    if y_np.ndim == 1:
        y_np = y_np[:, None]

    n, m = x_np.shape[0], y_np.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(x_np, y_np, metric='euclidean') ** k  # shape (n, m)
    wasserstein_cost = ot.emd2(a, b, M)
    return wasserstein_cost ** (1.0 / k)

