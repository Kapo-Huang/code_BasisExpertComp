import numpy as np
from torch.utils.data import Dataset


class NodePretrainDataset(Dataset):
    def __init__(self, dataset: Dataset, assignments: np.ndarray):
        if len(dataset) != int(len(assignments)):
            raise ValueError("assignments length must match dataset length")
        self.dataset = dataset
        self.assignments = np.asarray(assignments, dtype=np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        coords, _ = self.dataset[idx]
        return coords, int(self.assignments[idx])