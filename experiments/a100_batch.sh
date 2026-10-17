#!/usr/bin/env bash
# Jobs that need a 40 GB GPU: exact kernel baselines on all 50,000 images, then 50-epoch L=0 runs.
# Usage (from the repo root on the GPU machine): bash experiments/a100_batch.sh [out_dir] [50-epoch experiment names...]
set -u
OUT=${1:-results/a100}; shift || true
NAMES=${*:-gws_L0_T10_50ep}
mkdir -p "$OUT/logs"
python3 -m experiments.baselines --out "$OUT/baselines" >> "$OUT/logs/baselines.log" 2>&1 \
  || echo "baselines failed (exit $?)" >> "$OUT/logs/baselines.log"
python3 -m experiments.run experiments/configs/repro_50epoch.json --only $NAMES --source parquet --out "$OUT" \
  --skip-done >> "$OUT/logs/epoch50.log" 2>&1 || echo "epoch50 failed (exit $?)" >> "$OUT/logs/epoch50.log"
echo done > "$OUT/logs/ALL_DONE"
