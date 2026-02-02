#!/usr/bin/env bash
set -u

BASE_CONFIG="configs/BasisExpert/exp_data_ionization_light_basis_expert.yaml"
MAX_RUNS=9

# Per-run overrides (length >= MAX_RUNS). Adjust as needed.
BASE_DIM_LIST=(64 64 64 80 80 96 96 112 112)
BPT_LIST=(1 2 3 1 2 3 1 2 3)
LR_LIST=(5e-5 5e-5 5e-5 3e-5 3e-5 3e-5 2e-5 2e-5 2e-5)

LOG_ROOT="experiments/_batch_logs"
CFG_ROOT="experiments/_batch_configs"
mkdir -p "$LOG_ROOT" "$CFG_ROOT"

if [ ! -f "$BASE_CONFIG" ]; then
  echo "Base config not found: $BASE_CONFIG" >&2
  exit 1
fi

for ((i=0; i<MAX_RUNS; i++)); do
  run_idx=$((i+1))
  base_dim=${BASE_DIM_LIST[$i]}
  batches_per_timestep=${BPT_LIST[$i]}
  lr=${LR_LIST[$i]}

  cfg_out="$CFG_ROOT/run_${run_idx}.yaml"
  python - "$BASE_CONFIG" "$cfg_out" "$base_dim" "$batches_per_timestep" "$lr" <<'PY'
import sys
import yaml

base_path, out_path, base_dim, bpt, lr = sys.argv[1:]
with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("model", {})
cfg.setdefault("training", {})

cfg["model"]["base_dim"] = int(base_dim)
cfg["training"]["batches_per_timestep"] = int(bpt)
cfg["training"]["lr"] = float(lr)

with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  log_file="$LOG_ROOT/run_${run_idx}.log"
  {
    echo "=== Run ${run_idx}/${MAX_RUNS} ==="
    echo "Start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "Base config: $BASE_CONFIG"
    echo "Overrides: base_dim=$base_dim, batches_per_timestep=$batches_per_timestep, lr=$lr"
    echo "tqdm: disabled (INR_TQDM=0) because output is redirected to a log file"
    echo "Command: python -m inr.cli --config $cfg_out"
    echo "---- MODEL OUTPUT ----"
  } > "$log_file"

  INR_TQDM=0 python -m inr.cli --config "$cfg_out" 2>&1 | tee -a "$log_file"
  status=${PIPESTATUS[0]}
  echo "Exit code: $status" | tee -a "$log_file"
  echo "End: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "$log_file"

  if [ "$status" -ne 0 ]; then
    echo "Run ${run_idx} failed; continuing to next run." | tee -a "$log_file"
  fi

done

echo "All runs complete. Logs in $LOG_ROOT"