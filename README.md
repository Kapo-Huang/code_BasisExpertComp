# ScientificCompressionWithINR (PyTorch)

Modularized PyTorch project for the three INR experiments (Exp1/3/4). Uses SIREN and SIREN-ResNet variants with normalized datasets, validation, checkpointing, and batched prediction.

## Quick start

```bash
# create/activate your environment first
pip install -r requirements.txt

# run any preset
PYTHONPATH=src python -m inr.cli --config configs/exp1.yaml
PYTHONPATH=src python -m inr.cli --config configs/exp3.yaml
PYTHONPATH=src python -m inr.cli --config configs/exp4_resnet.yaml
PYTHONPATH=src python -m inr.cli --config configs/exp6.yaml  # MoE-INR on XYZT->E
```

On PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m inr.cli --config configs/exp4_resnet.yaml
```

Outputs and checkpoints are written under `outputs/` by default. Override paths by editing the YAML config.

## Project layout

- `configs/exp*.yaml` — data paths, model hyperparameters, training settings.
- `src/inr/data.py` — reusable `NodeDataset` with normalization + denorm helpers.
- `src/inr/models/` — `siren.py` (vanilla) and `siren_resnet.py` (Exp4 architecture).
- `src/inr/training/loops.py` — train/val loop, checkpointing, full-field prediction.
- `src/inr/utils/io.py` — checkpoint utilities.
- `src/inr/cli.py` — CLI entrypoint that loads a config, builds dataset/model, trains and predicts.

## Config fields

```yaml
data:
  x_path: ./data/data_XYZ.npy   # input coordinates (N,3) or (N,4)
  y_path: ./data/data_U.npy     # targets (N,3) or (N,24)
  normalize: true

model:
  name: siren | siren_resnet
  # siren-specific:
  in_features: 3
  out_features: 3
  hidden_features: 512
  hidden_layers: 5
  first_omega_0: 30.0
  hidden_omega_0: 30.0

training:
  epochs: 500
  batch_size: 131072
  pred_batch_size: 131072
  lr: 5e-5
  val_split: 0.1
  log_every: 4
  save_every: 10
  early_stop_patience: 0
  seed: 42
  save_model: outputs/siren_exp1.pth
  save_pred: outputs/predicted_U.npy
```

## Notes

- The normalization stats are stored with each checkpoint and reused for prediction.
- Validation split is capped to 50% to avoid degenerate splits.
- `pred_batch_size` is used to avoid OOM during full-field prediction.


zhy git test