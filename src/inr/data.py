from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NodeDataset(Dataset):
    """
    Loads paired coordinate/target tensors with optional normalization.

    Preferred on-disk layout:
    - raw:       data/raw/<dataset>/<split>/{coords.npy,targets.npy,conn.npy}
    - processed: data/processed/<dataset>/<version>/<split>/{coords.npy,targets.npy}
    Paths can still be overridden explicitly via x_path/y_path.
    """

    def __init__(
        self,
        x_path: str,
        y_path: str,
        normalize: bool = True,
        stats_path: Optional[str] = None,
        eps: float = 1e-12,
    ):
        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched samples: x={x.shape[0]} y={y.shape[0]}")

        # self.x = torch.from_numpy(x.astype(np.float32))
        # self.y = torch.from_numpy(y.astype(np.float32))
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.normalize = normalize
        self.stats_path = stats_path
        self.eps = float(eps)

        if normalize:
            stats_loaded = False
            if stats_path and Path(stats_path).exists():
                stats = np.load(stats_path)
                try:
                    self.x_mean = torch.from_numpy(stats["x_mean"]).to(torch.float32)
                    self.x_std = torch.from_numpy(stats["x_std"]).to(torch.float32)
                    self.y_mean = torch.from_numpy(stats["y_mean"]).to(torch.float32)
                    self.y_std = torch.from_numpy(stats["y_std"]).to(torch.float32)
                    self.x_std = torch.where(self.x_std == 0, torch.ones_like(self.x_std), self.x_std)
                    self.y_std = torch.where(self.y_std == 0, torch.ones_like(self.y_std), self.y_std)
                    stats_loaded = True
                except KeyError:
                    stats_loaded = False

            if not stats_loaded:
                self.x_mean, self.x_std = self._compute_stats(self.x)
                self.y_mean, self.y_std = self._compute_stats(self.y)
                if stats_path:
                    self._save_stats(stats_path)
            self.x = (self.x - self.x_mean) / self.x_std
            self.y = (self.y - self.y_mean) / self.y_std
        else:
            self.x_mean = torch.zeros_like(self.x[:1])
            self.x_std = torch.ones_like(self.x[:1])
            self.y_mean = torch.zeros_like(self.y[:1])
            self.y_std = torch.ones_like(self.y[:1])

    @staticmethod
    def _compute_stats(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = tensor.mean(0, keepdim=True)
        std = tensor.std(0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return mean, std

    def _save_stats(self, stats_path: str) -> None:
        Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            stats_path,
            x_mean=self.x_mean.numpy(),
            x_std=self.x_std.numpy(),
            y_mean=self.y_mean.numpy(),
            y_std=self.y_std.numpy(),
        )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        xb = self.x[idx]
        yb = self.y[idx]
        return xb, yb

    def denormalize_targets(self, y_norm: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return y_norm
        return y_norm * self.y_std.to(y_norm.device) + self.y_mean.to(y_norm.device)

    def input_stats(self):
        return {"mean": self.x_mean.numpy(), "std": self.x_std.numpy()}

    def target_stats(self):
        return {"mean": self.y_mean.numpy(), "std": self.y_std.numpy()}
