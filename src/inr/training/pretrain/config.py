from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PretrainConfig:
    enabled: bool = False
    epochs: int = 0
    lr: float = 5e-5
    batch_size: int = 8000
    cluster_num_time_samples: int = 16
    cluster_seed: int = 42
    assignments_cache_path: str = ""
    assignments_method: str = "voxel_clustering"
    spatial_blocks: Optional[Tuple[int, int, int]] = None
    time_block_size: int = 0
    mode: str = "router_classification"