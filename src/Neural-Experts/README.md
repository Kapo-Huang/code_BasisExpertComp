# Neural-Experts Ionization

This directory has been reduced to a single task: 4D ionization INR training with the Neural-Experts MoE architecture.

Supported attributes:

- `GT`
- `H+`
- `H2`
- `He`
- `PD`

Each run trains one attribute at a time with input `(x, y, z, t)` and output `v`.

## Entry Points

Manager pretraining:

```powershell
python Neural-Experts\ionization\train_ionization.py --config Neural-Experts\configs\ionization\config_ionization_Hplus_managerpretraining.yaml --identifier hplus_managerpretrain --logdir experiments\neural_experts --gpu 0
```

Main training:

```powershell
python Neural-Experts\ionization\train_ionization.py --config Neural-Experts\configs\ionization\config_ionization_Hplus.yaml --identifier hplus_main --logdir experiments\neural_experts --gpu 0
```

Validation export:

```powershell
python validate_prediction.py --config experiments\neural_experts\hplus_main\validate_artifacts\config.yaml --checkpoint experiments\neural_experts\hplus_main\validate_artifacts\hplus_main.pth --timestamp 0 --batch-size 262144 --outdir validate_out\hplus_pred --prefix hplus --device cuda
```

## Scope

Retained components:

- ionization dataset and configs
- Neural-Experts INR / INR_MoE model code
- random-balanced segmentation pretraining
- ionization loss and stage handling
- validate-compatible checkpoint export

Removed components:

- legacy task-specific training code and sample assets
