#!/usr/bin/env bash
# Runs the experiments for the claims still open in docs/REPRODUCTION.md on one GPU with >= 90 GB (RTX PRO 6000 /
# H100 NVL class). Two streams: A holds the large tables one at a time (GWS L = 2 at 50 epochs needs ~84 GB);
# B runs the 10-block DeltaNet stack (~3 GB) beside it. Logs: $OUT/*.log; markers: $OUT/A_DONE, $OUT/B_DONE.
# Usage: bash experiments/queue_open_claims.sh [OUT=results/open] [DATA=data]
set -u
OUT=${1:-results/open}
DATA=${2:-data}
mkdir -p "$OUT"
export RCDL_TF32=1 RCDL_FUSED_ADAM=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="python3 -m experiments.run --data $DATA --skip-done"

stream_a() {
  # claim 11: bf16 check at 10 epochs (fp32: 65.05%), then L = 2 at 50 epochs on CIFAR-10 and CIFAR-100
  $R experiments/configs/multi_epoch_bf16.json --only gws_L2_T10_10ep_bf16 gws_L2_T10_50ep_bf16 --out "$OUT/multi"
  $R experiments/configs/multi_epoch_cifar100.json --only gws_L2_T10_50ep_c100_bf16 --out "$OUT/multi"
  # claim 15: Fig. 5 at 1/10 of the data with the paper's ratios
  $R experiments/configs/overfit_fig5_scaled.json --out "$OUT/fig5"
  # claim 17: the paper's per-threshold tuning at tau = 35 (the notebook setting T5_b30_lr0.001 was run before)
  NAMES=$(python3 -c "import json; print(' '.join(k for k in json.load(open('experiments/configs/hybrid_tau35_tuning.json'))['experiments'] if k != 'hybrid_tau35_T5_b30_lr0.001'))")
  $R experiments/configs/hybrid_tau35_tuning.json --only $NAMES --out "$OUT/hybrid35"
  touch "$OUT/A_DONE"
}

stream_b() {
  # claim 18: DeltaNet decay, 10-block residual MLP, 2,000 epochs, the paper's five decay rates in one stack
  python3 -m experiments.deltanet_long --out "$OUT/deltanet" --data "$DATA" --epochs 2000 --lams 0 7e-6 1e-5 2e-5 3e-5
  touch "$OUT/B_DONE"
}

stream_a > "$OUT/stream_a.log" 2>&1 &
stream_b > "$OUT/stream_b.log" 2>&1 &
wait
