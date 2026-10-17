#!/usr/bin/env bash
# Single-pass CIFAR-10 reproductions and ablations on one GPU (written for a Colab L4, 22 GB).
# Usage (from the repo root on the GPU machine): bash experiments/gpu_batch.sh [out_dir] [seeds...]
# Each batch resumes with --skip-done; memory-heavy runs (psm100 ~13 GB, hybrid ~18 GB) come last.
set -u
OUT=${1:-results/gpu}; shift || true
SEEDS=${*:-1 2 3 4 5}
mkdir -p "$OUT/logs"
run() {  # run <config> <log name> <experiment names...>
  local cfg=$1 log=$2; shift 2
  python3 -m experiments.run "$cfg" --only "$@" --seeds $SEEDS --source parquet --out "$OUT" --skip-done \
    >> "$OUT/logs/$log.log" 2>&1 || echo "batch $log failed (exit $?)" >> "$OUT/logs/$log.log"
}
run experiments/configs/repro_1epoch.json repro_core gws rbf gauge mlp_adam mlp_muon
run experiments/configs/ablations_1epoch.json ablations gws_eta0 gws_noclip gws_eta0p1 rbf_T6 gws_L0 gws_L1
run experiments/configs/repro_1epoch.json repro_rest mlp_sgd mlp_rmsprop psm10
run experiments/configs/repro_1epoch.json repro_heavy psm100 hybrid
echo done > "$OUT/logs/ALL_DONE"
