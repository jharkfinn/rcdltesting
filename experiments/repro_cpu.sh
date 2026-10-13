#!/usr/bin/env bash
# Sequential CPU reproduction of the notebook's single-pass CIFAR-10 runs (about 12 min per growing-layer run on
# 4 Xeon cores). Usage: bash experiments/repro_cpu.sh <source: parquet|binary> <out_dir> <name:seeds> ...
#   e.g. bash experiments/repro_cpu.sh parquet results/hf "gws:1 2 3" "rbf:1 2 3"
set -u
SOURCE=$1; OUT=$2; shift 2
mkdir -p "$OUT/logs"
for spec in "$@"; do
  name=${spec%%:*}; seeds=${spec#*:}
  for s in $seeds; do
    python3 -m experiments.run experiments/configs/repro_1epoch.json --only "$name" --seeds "$s" \
      --source "$SOURCE" --out "$OUT" --skip-done > "$OUT/logs/${name}_seed${s}.log" 2>&1
    tail -n 1 "$OUT/logs/${name}_seed${s}.log"
  done
done
