# Neural-Experts Mesh

This directory is now Mesh-only. `Ionization` volume training has been removed.

Supported datasets:

- `ocean`
  - `fort63`
  - `fort64`
  - `fort73`
  - `speed`
  - `v`
- `stress`
  - `cell_E`
  - `cell_E_IntegrationPoints`
  - `cell_S`
  - `cell_S_IntegrationPoints`
  - `point_RF`
  - `point_U`

Each run trains one attribute at a time with input `(x, y, z, t)`.

## Entry Points

Manager pretraining:

```powershell
python Neural-Experts\mesh\train_mesh.py --config Neural-Experts\configs\stress\config_stress_cell_E_managerpretraining.yaml --identifier stress_cell_E_managerpretrain --logdir experiments\neural_experts_mesh --gpu 0
```

Main training:

```powershell
python Neural-Experts\mesh\train_mesh.py --config Neural-Experts\configs\stress\config_stress_cell_E.yaml --identifier stress_cell_E_main --logdir experiments\neural_experts_mesh --gpu 0
```

Prediction export:

```powershell
python Neural-Experts\predict_mesh.py --config experiments\neural_experts_mesh\stress_cell_E_main\validate_artifacts\config.yaml --checkpoint experiments\neural_experts_mesh\stress_cell_E_main\validate_artifacts\stress_cell_E_main.pth --timestamps 0 --batch-size 16000 --device cuda
```

PSNR validation:

```powershell
python Neural-Experts\validate_mesh_psnr.py --config experiments\neural_experts_mesh\stress_cell_E_main\validate_artifacts\config.yaml --checkpoint experiments\neural_experts_mesh\stress_cell_E_main\validate_artifacts\stress_cell_E_main.pth --timestamps 0 --batch-size 16000 --device cuda
```

## Config Layout

- `configs/ocean/`: one `*_managerpretraining.yaml` and one main `.yaml` per attribute
- `configs/stress/`: one `*_managerpretraining.yaml` and one main `.yaml` per attribute

Important config fields:

- `DATA.source_path`
- `DATA.target_path`
- `DATA.target_stats_path`
- `DATA.stats_key`
- `DATA.association`
- `TRAINING.pretrain_assignment.method`
- `TRAINING.pretrain_assignment.fit_samples`
- `TRAINING.pretrain_assignment.cache_path`

## Output Layout

Each training run writes:

- `config.yaml`: resolved Mesh config used for the run
- `trained_models/`: raw training checkpoints
- `validate_artifacts/config.yaml`: resolved Mesh config for prediction / validation
- `validate_artifacts/<identifier>.pth`: checkpoint with model weights and normalization stats

## Notes

- `ocean` uses shared point coordinates from `source_XYZT.npy`
- `stress` switches between `source_cell_XYZT.npy` and `source_point_XYZT.npy` by attribute
- Router pretraining uses coordinate KMeans pseudo-labels instead of voxel patch segmentation
