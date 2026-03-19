from torch.utils.data import DataLoader

def build_dataset(cfg, file_id, training=True):
    if cfg['DATA'].get('name') == 'RGBINR':
        from .RGBImages import RGBINRDataset
        dataset = RGBINRDataset(cfg['DATA'], file_id)
    elif cfg['DATA'].get('name') == 'sdf_3d':
        from .dataset3dSDF import SDFdataset3D
        dataset = SDFdataset3D(cfg, file_id)
    elif cfg['DATA'].get('name') == 'Audio':
        from .Audio import AudioINRDataset
        dataset = AudioINRDataset(cfg['DATA'], file_id)
    elif cfg['DATA'].get('name') == 'ionization':
        from .Ionization import IonizationINRDataset
        dataset = IonizationINRDataset(cfg, file_id)
    else:
        raise NotImplementedError
    return dataset


def build_dataloader(cfg, file_id, training=True):
    split = 'TRAINING' if training else 'TESTING'
    dataset = build_dataset(cfg, file_id, training)
    dataloader = DataLoader(dataset=dataset, num_workers=int(cfg[split].get('num_workers', 0)), pin_memory=False,
                            batch_size=cfg[split]['batch_size'], shuffle=False)

    return dataloader, dataset
