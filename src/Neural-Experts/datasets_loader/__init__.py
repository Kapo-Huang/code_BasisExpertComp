from torch.utils.data import DataLoader

def build_dataset(cfg, file_id, training=True):
    from .Mesh import MeshAttributeDataset

    dataset = MeshAttributeDataset(cfg, file_id)
    return dataset


def build_dataloader(cfg, file_id, training=True):
    split = 'TRAINING' if training else 'TESTING'
    dataset = build_dataset(cfg, file_id, training)
    dataloader = DataLoader(dataset=dataset, num_workers=int(cfg[split].get('num_workers', 0)), pin_memory=False,
                            batch_size=cfg[split]['batch_size'], shuffle=False)

    return dataloader, dataset
