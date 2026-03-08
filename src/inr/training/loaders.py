from torch.utils.data import DataLoader


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, sampler=None) -> DataLoader:
    kwargs = {
        "pin_memory": True,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        **kwargs,
    )