#!/usr/bin/env bash
# E6 umbrella = bee (protocol v2, the critique's revised protocol) on one molab GPU: setup, then every arm as a queue
# with up to $WORKERS concurrent jobs. Stages (each waits for the previous one):
#   0  setup: code, validation set, packed training set, E1 (and E1r) files from the HF results dataset, hf_sync of the
#      runs, selection.json, arm-E selection (rng 2026), evaluation sets, provenance tables
#   A  lr selection: P1 as selected, seed 0, frozen-v, lr in {1e-4, 3e-4, 1e-3, 3e-3}, taught and untaught (each lr
#      its own warm-up); lr_select applies the pre-registered rule to tau.json and the untaught drift on V_ref ->
#      lr_star.json
#   1  photos re-verified at every branch point (a photo no longer top-1 for the source at theta_w is replaced by the
#      next reserve photo; recorded in photos_*.json), then taught and solo runs at lr*: B (P1-P5 x seeds 0-2), Bp
#      (standard optimizer, seed 0), C (solo, frozen-v and standard), F (controls noise:0, noise:1, shuffled P1, P2 at
#      seed 0; noise:0, shuffled P1 at seeds 1, 2; solo of the seed-0 ones), E (6 relabel triples), S2 (secondary
#      weak-route arm source_train_images, seed 0, with controls)
#   2  untaught runs, saving at every tau and tau + 500 of the runs they serve (a finished untaught run that misses one
#      of them is redone)
#   3  analyses of every taught and solo run; arm I leave-group-out continuations (B, seeds 0 and 1, 4 variants)
#   4  arm J, only when E1r's final files exist: E1r selection (photos passing in both models first), taught and
#      untaught at seed 0, analyses
#   5  decision rules over every analysis -> decisions.json (always recomputed)
# Every job has a DONE marker ($U/logs/<job>.done); a rerun skips finished jobs and repeats unfinished ones from the
# start (the fine-tuning script has no mid-run resume; an unfinished run's directory is cleared first, see ft).
# Failures are listed in $U/logs/FAILED.
# Needs /marimo/work/.hf_token. Usage: bash experiments/molab_umbrella.sh
#   env: WORKERS (3), W (/marimo/work), U ($W/umb), CODE ($U/code), STAGES ("A 1 2 3 4 5"), SECONDARY (1), LRS, SEEDS
set -uo pipefail
W=${W:-/marimo/work}
U=${U:-$W/umb}
CODE=${CODE:-$U/code}
GIT_BRANCH=${GIT_BRANCH:-claude/rcdl-repro}
E1=${E1:-$W/results/e1_tiny_adamw}
E1R=${E1R:-$W/results/e1r_tiny_adamw_s1}
VAL=$W/data/imagenet/data
TRAIN=${TRAIN:-/dev/shm/imagenet256}
WORKERS=${WORKERS:-3}
STAGES=${STAGES:-"A 1 2 3 4 5"}
SECONDARY=${SECONDARY:-1}
LRS=${LRS:-"1e-4 3e-4 1e-3 3e-3"}
SEEDS=${SEEDS:-"0 1 2"}
ARM=strong_source                 # primary (decided before any fine-tune: E1 has no umbrella route)
ARM2=source_train_images          # secondary, weak route (umbrella training images)
HFREPO=joshhf/rcdl-molab-results
R=$U/runs; L=$U/logs; C=$U/cache; BR=$U/branches
SEL=$U/selection.json; SELE=$U/selection_E.json
mkdir -p $R $L $C $BR
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -d $CODE/.git ] || git clone -q --depth 1 -b $GIT_BRANCH https://github.com/jharkfinn/rcdltesting $CODE
cd $CODE && git log --oneline -1
UMB="python3 -m experiments.convnext_umbrella"
ANA="python3 -m experiments.convnext_umbrella_analysis"
ARMS="python3 -m experiments.convnext_umbrella_arms"
has() { [[ " $STAGES " == *" $1 "* ]]; }

# job <name> <command...>: run in the background, at most $WORKERS at once, unless <name> is done
job() {
  local name=$1; shift
  [ -f "$L/$name.done" ] && return 0
  while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do wait -n || true; done
  echo "$(date +%T) start $name"
  ( if "$@" > "$L/$name.log" 2>&1; then touch "$L/$name.done"; echo "$(date +%T) done $name";
    else echo "$(date +%T) $name failed (see $L/$name.log)" | tee -a "$L/FAILED"; fi ) &
}
barrier() { wait; }
# ft <job name> <convnext_umbrella run command>: a fine-tune job. The directory (--out) of a run that did not finish is
# cleared first: the script restarts from step 0, and files of the interrupted attempt at steps the new attempt does not
# save (its tau and tau + 500, if tau differs) would otherwise become vertices of the analysis paths
ft() {
  local name=$1 out= prev= x; shift
  for x in "$@"; do [ "$prev" = --out ] && out=$x; prev=$x; done
  [ -n "$out" ] && [ ! -f "$L/$name.done" ] && [ ! -f "$out/DONE" ] && rm -rf "$out"
  job "$name" "$@"
}
# shuffled control of a photo spec
shuf_of() { case $1 in val:*) echo "shuffle:${1#val:}";; train:*) echo "shuffle_train:${1#train:}";; esac; }
# an untaught run that is done but lacks some of the needed saves is removed, so that it is redone
stale_untaught() {   # stale_untaught <job name> <run dir> <full saves>
  [ -f "$L/$1.done" ] || return 0
  python3 -c "import json,sys; c=json.load(open(sys.argv[1]+'/config.json')); need=set(filter(None,sys.argv[2].split(',')));\
have=set(filter(None,c['full_saves'].split(','))); sys.exit(0 if need <= have else 1)" "$2" "$3" \
    || { echo "redo $1 (missing saves)"; rm -rf "$2" "$L/$1.done"; }
}

# ---------------------------------------------------------------------------------------------------- 0 setup
python3 - <<EOF
import glob, os
from huggingface_hub import hf_hub_download, snapshot_download
tok = open('$W/.hf_token').read().strip()
if not glob.glob('$VAL/validation-*.parquet'):
  snapshot_download('ILSVRC/imagenet-1k', repo_type='dataset', token=tok, local_dir='$W/data/imagenet',
                    allow_patterns=['classes.py', 'data/validation-*'])
  print('validation set downloaded', flush=True)
files = ['config.json', 'model_25020.pt', 'attr_25020.pt', 'attr_init.pt', 'atlas_25020.pt', 'history/snap_25020.pt']
for run, dst_dir in (('e1_tiny_adamw', '$E1'), ('e1r_tiny_adamw_s1', '$E1R')):
  for f in files:
    dst = os.path.join(dst_dir, f)
    if os.path.exists(dst):
      continue
    try:
      p = hf_hub_download('$HFREPO', f'neuron-dev/{run}/{f}', repo_type='dataset', token=tok, local_dir='$W/hf_restore')
      os.makedirs(os.path.dirname(dst), exist_ok=True)
      os.replace(p, dst)
      print('fetched', run, f, flush=True)
    except Exception as e:
      print('not available:', run, f, type(e).__name__, flush=True)
EOF
for f in model_25020.pt attr_25020.pt attr_init.pt history/snap_25020.pt; do
  [ -f $E1/$f ] || { echo "E1 file $E1/$f missing: stop"; exit 1; }
done
[ -f $E1/atlas_25020.pt ] || echo "WARNING: $E1/atlas_25020.pt missing: M3 runs without pole sets"
if [ ! -f $TRAIN/DONE ]; then
  python3 -c "from rcdl.imagenet import prepare_train; prepare_train(token=open('$W/.hf_token').read().strip())" \
    > $L/prep_train.log 2>&1
fi
if ! pgrep -f "hf_sync.py --root $R" > /dev/null; then
  setsid nohup python3 experiments/hf_sync.py --root $R --prefix neuron-dev/umb --every 600 > $L/hf_sync.log 2>&1 &
  disown
fi
[ -f $SEL ] || $UMB select --run $E1 --val $VAL --out $SEL > $L/select.log 2>&1 || { echo "selection failed"; exit 1; }
[ -f $SELE ] || $ARMS select_e --run $E1 --sel $SEL --val $VAL --out $SELE > $L/select_e.log 2>&1 \
  || { echo "arm-E selection failed"; exit 1; }
mapfile -t P < <($ARMS photos --sel $SEL --arm $ARM)
[ ${#P[@]} -ge 3 ] || { echo "fewer than 3 photos in arm $ARM"; exit 1; }
NP=${#P[@]}
HAS2=$(python3 -c "import json; print(int('$ARM2' in json.load(open('$SEL'))['arms']))")
[ "$SECONDARY" = 1 ] && [ "$HAS2" = 1 ] && mapfile -t P2 < <($ARMS photos --sel $SEL --arm $ARM2) || P2=()
job vsets_$ARM $ANA vsets --first10 $C/first10.pt --sel $SEL --sel $SELE --arm $ARM --val $VAL --out $C/vsets_$ARM.pt
job prov_$ARM $ANA provenance --sel $SEL --arm $ARM --e1 $E1 --val $VAL --out $C/prov_$ARM.pt
barrier                                            # the image caches ($C/first10.pt, /dev/shm/probe_8.pt) exist now
VS="$ANA vsets --first10 $C/first10.pt --sel $SEL --sel $SELE --val $VAL"
for i in 0 1 2 3 4 5; do job vsets_E$i $VS --arm E$i --out $C/vsets_E$i.pt; done
[ ${#P2[@]} -gt 0 ] && job vsets_$ARM2 $VS --arm $ARM2 --out $C/vsets_$ARM2.pt
[ ${#P2[@]} -gt 0 ] && job prov_$ARM2 $ANA provenance --sel $SEL --arm $ARM2 --e1 $E1 --val $VAL --out $C/prov_$ARM2.pt
barrier

UARGS="--snapshot $E1/history/snap_25020.pt --val $VAL --train $TRAIN --branch_dir $BR"
RUN="$UMB run $UARGS"

# --------------------------------------------------------------------------------------------- A lr selection
if has A; then
  for lr in $LRS; do
    ft A_lr${lr}_taught $RUN --sel $SEL --arm $ARM --mode taught --photo ${P[0]} --lr $lr --seed 0 \
      --out $R/A/lr$lr/taught
    ft A_lr${lr}_untaught $RUN --sel $SEL --arm $ARM --mode untaught --lr $lr --seed 0 --out $R/A/lr$lr/untaught
  done
  barrier
  job A_select $ARMS lr_select --runs $R/A --e1 $E1 --vsets $C/vsets_$ARM.pt --out $U/lr_star.json
  barrier
fi
LR=$([ -f $U/lr_star.json ] && $ARMS lr_star --file $U/lr_star.json)
[ -n "$LR" ] || { echo "no lr* (see $U/lr_star.json): stop"; exit 1; }
echo "lr* = $LR"

# ------------------------------------------------------------------------------------ 1 taught and solo runs
# controls use the pre-registered P1 and P2 of the selection; photos are re-verified at every branch point
declare -A CTRL=([noise0]=noise:0 [noise1]=noise:1 [shufP1]=$(shuf_of ${P[0]}) [shufP2]=$(shuf_of ${P[1]}))
if has 1; then
  for sd in $SEEDS; do
    job verify_s$sd $ARMS verify --n $NP --out $U/photos_${ARM}_s$sd.json -- $UARGS --sel $SEL --arm $ARM \
      --mode untaught --lr $LR --seed $sd
  done
  for i in 0 1 2 3 4 5; do
    job verify_E$i $ARMS verify --n 1 --out $U/photos_E$i.json -- $UARGS --sel $SELE --arm E$i --mode untaught \
      --lr $LR --seed 0
  done
  [ ${#P2[@]} -gt 0 ] && job verify_S2 $ARMS verify --n ${#P2[@]} --out $U/photos_${ARM2}_s0.json -- $UARGS \
    --sel $SEL --arm $ARM2 --mode untaught --lr $LR --seed 0
  barrier
  for sd in $SEEDS; do
    mapfile -t PS < <($ARMS photos --verified $U/photos_${ARM}_s$sd.json)
    for ((i = 0; i < ${#PS[@]}; i++)); do
      ft B_s${sd}_P$((i + 1)) $RUN --sel $SEL --arm $ARM --mode taught --photo ${PS[$i]} --lr $LR --seed $sd \
        --out $R/B/s$sd/P$((i + 1))
    done
    for c in $([ $sd = 0 ] && echo noise0 noise1 shufP1 shufP2 || echo noise0 shufP1); do
      ft F_s${sd}_$c $RUN --sel $SEL --arm $ARM --mode taught --photo ${CTRL[$c]} --lr $LR --seed $sd --out $R/F/s$sd/$c
    done
  done
  mapfile -t PS < <($ARMS photos --verified $U/photos_${ARM}_s0.json)
  for ((i = 0; i < ${#PS[@]}; i++)); do
    ft Bp_P$((i + 1)) $RUN --sel $SEL --arm $ARM --opt standard --mode taught --photo ${PS[$i]} --lr $LR --seed 0 \
      --out $R/Bp/P$((i + 1))
    for o in frozen_v standard; do
      ft C_${o}_P$((i + 1)) $RUN --sel $SEL --arm $ARM --opt $o --mode solo --photo ${PS[$i]} --lr $LR --seed 0 \
        --out $R/C/${o}_P$((i + 1))
    done
  done
  for c in noise0 noise1 shufP1 shufP2; do
    ft Fsolo_$c $RUN --sel $SEL --arm $ARM --mode solo --photo ${CTRL[$c]} --lr $LR --seed 0 --out $R/F/solo/$c
  done
  for i in 0 1 2 3 4 5; do
    EP=$($ARMS photos --verified $U/photos_E$i.json --n 1)
    [ -n "$EP" ] && ft E${i}_taught $RUN --sel $SELE --arm E$i --mode taught --photo $EP --lr $LR --seed 0 \
      --out $R/E/E$i/taught
  done
  if [ ${#P2[@]} -gt 0 ]; then
    mapfile -t PS < <($ARMS photos --verified $U/photos_${ARM2}_s0.json)
    for ((i = 0; i < ${#PS[@]}; i++)); do
      ft S2_P$((i + 1)) $RUN --sel $SEL --arm $ARM2 --mode taught --photo ${PS[$i]} --lr $LR --seed 0 \
        --out $R/S2/P$((i + 1))
    done
    declare -A CTRL2=([noise0]=noise:0 [noise1]=noise:1 [shufP1]=$(shuf_of ${P2[0]}) [shufP2]=$(shuf_of ${P2[1]}))
    for c in noise0 noise1 shufP1 shufP2; do
      ft S2F_$c $RUN --sel $SEL --arm $ARM2 --mode taught --photo ${CTRL2[$c]} --lr $LR --seed 0 --out $R/S2/F/$c
    done
  fi
  barrier
fi

# ------------------------------------------------------------------------------------------- 2 untaught runs
if has 2; then
  for sd in $SEEDS; do
    FS=$($ARMS full_saves $R/B/s$sd/P* $R/F/s$sd/*)
    stale_untaught B_s${sd}_untaught $R/B/s$sd/untaught "$FS"
    ft B_s${sd}_untaught $RUN --sel $SEL --arm $ARM --mode untaught --lr $LR --seed $sd --full_saves "$FS" \
      --out $R/B/s$sd/untaught
  done
  FS=$($ARMS full_saves $R/Bp/P*)
  stale_untaught Bp_untaught $R/Bp/untaught "$FS"
  ft Bp_untaught $RUN --sel $SEL --arm $ARM --opt standard --mode untaught --lr $LR --seed 0 --full_saves "$FS" \
    --out $R/Bp/untaught
  for i in 0 1 2 3 4 5; do
    [ -d $R/E/E$i/taught ] || continue
    FS=$($ARMS full_saves $R/E/E$i/taught)
    stale_untaught E${i}_untaught $R/E/E$i/untaught "$FS"
    ft E${i}_untaught $RUN --sel $SELE --arm E$i --mode untaught --lr $LR --seed 0 --full_saves "$FS" \
      --out $R/E/E$i/untaught
  done
  if [ ${#P2[@]} -gt 0 ]; then
    FS=$($ARMS full_saves $R/S2/P* $R/S2/F/*)
    stale_untaught S2_untaught $R/S2/untaught "$FS"
    ft S2_untaught $RUN --sel $SEL --arm $ARM2 --mode untaught --lr $LR --seed 0 --full_saves "$FS" --out $R/S2/untaught
  fi
  barrier
fi

# -------------------------------------------------------------------------------- 3 analyses and arm I
AN="$ANA run --val $VAL --train $TRAIN --e1 $E1"
# runs <glob>: the finished run directories matching it
runs() { for d in $1; do [ -f $d/DONE ] && echo ${d%/}; done; }
if has 3; then
  for sd in $SEEDS; do
    for d in $(runs "$R/B/s$sd/P*/"); do
      job an_B_s${sd}_$(basename $d) $AN --run $d --untaught $R/B/s$sd/untaught --tag B --vsets $C/vsets_$ARM.pt \
        --provenance $C/prov_$ARM.pt
    done
    for d in $(runs "$R/F/s$sd/*/"); do
      job an_F_s${sd}_$(basename $d) $AN --run $d --untaught $R/B/s$sd/untaught --tag F --vsets $C/vsets_$ARM.pt
    done
  done
  for d in $(runs "$R/Bp/P*/"); do
    job an_Bp_$(basename $d) $AN --run $d --untaught $R/Bp/untaught --tag Bp --vsets $C/vsets_$ARM.pt --skip M3
  done
  for d in $(runs "$R/C/*/"); do
    job an_C_$(basename $d) $AN --run $d --tag C --vsets $C/vsets_$ARM.pt
  done
  for d in $(runs "$R/F/solo/*/"); do
    job an_Fsolo_$(basename $d) $AN --run $d --tag Fsolo --vsets $C/vsets_$ARM.pt
  done
  for i in 0 1 2 3 4 5; do
    [ -f $R/E/E$i/taught/DONE ] && job an_E$i $AN --run $R/E/E$i/taught --untaught $R/E/E$i/untaught --tag E \
      --vsets $C/vsets_E$i.pt
  done
  if [ ${#P2[@]} -gt 0 ]; then
    for d in $(runs "$R/S2/P*/"); do
      job an_S2_$(basename $d) $AN --run $d --untaught $R/S2/untaught --tag S2 --vsets $C/vsets_$ARM2.pt \
        --provenance $C/prov_$ARM2.pt
    done
    for d in $(runs "$R/S2/F/*/"); do
      job an_S2F_$(basename $d) $AN --run $d --untaught $R/S2/untaught --tag S2F --vsets $C/vsets_$ARM2.pt --skip M3
    done
  fi
  for sd in 0 1; do
    for d in $(runs "$R/B/s$sd/P*/"); do
      for v in base target source A; do
        job I_s${sd}_$(basename $d)_$v $ARMS lgo --from_run $d --variant $v --val $VAL --train $TRAIN \
          --out $R/I/s$sd/$(basename $d)/$v
      done
    done
  done
  barrier
fi

# ------------------------------------------------------------------------------------------- 4 arm J (E1r)
if has 4; then
  if [ -f $E1R/model_25020.pt ] && [ -f $E1R/history/snap_25020.pt ]; then
    SELJ=$U/selection_J.json
    [ -f $U/selection_E1r.json ] || $UMB select --run $E1R --val $VAL --out $U/selection_E1r.json > $L/select_J.log 2>&1
    [ -f $SELJ ] || $ARMS select_j --sel_r $U/selection_E1r.json --sel_e1 $SEL --out $SELJ >> $L/select_J.log 2>&1
    JARM=$(python3 -c "import json; a=json.load(open('$SELJ'))['arms']; print('$ARM' if '$ARM' in a else next(iter(a)))")
    UARGSJ="--snapshot $E1R/history/snap_25020.pt --val $VAL --train $TRAIN --branch_dir $BR"
    RUNJ="$UMB run $UARGSJ"
    job vsets_J $ANA vsets --first10 $C/first10.pt --sel $SELJ --sel $SEL --sel $SELE --arm $JARM --val $VAL \
      --out $C/vsets_J.pt
    job verify_J $ARMS verify --n 5 --out $U/photos_J_s0.json -- $UARGSJ --sel $SELJ --arm $JARM --mode untaught \
      --lr $LR --seed 0
    barrier
    mapfile -t PJ < <($ARMS photos --verified $U/photos_J_s0.json)
    for ((i = 0; i < ${#PJ[@]}; i++)); do
      ft J_P$((i + 1)) $RUNJ --sel $SELJ --arm $JARM --mode taught --photo ${PJ[$i]} --lr $LR --seed 0 \
        --out $R/J/P$((i + 1))
    done
    barrier
    FS=$($ARMS full_saves $R/J/P*)
    stale_untaught J_untaught $R/J/untaught "$FS"
    ft J_untaught $RUNJ --sel $SELJ --arm $JARM --mode untaught --lr $LR --seed 0 --full_saves "$FS" --out $R/J/untaught
    barrier
    for d in $(runs "$R/J/P*/"); do
      job an_J_$(basename $d) $ANA run --val $VAL --train $TRAIN --e1 $E1R --run $d --untaught $R/J/untaught --tag J \
        --vsets $C/vsets_J.pt --skip M3
    done
    barrier
  else
    echo "arm J skipped: E1r's final files ($E1R/model_25020.pt, history/snap_25020.pt) are not available yet"
  fi
fi

# -------------------------------------------------------------------------------------------- 5 decisions
if has 5; then
  $ANA decide --root $R --out $U/decisions.json > $L/decide.log 2>&1 && tail -1 $L/decide.log
fi
[ -f $L/FAILED ] && { echo "failed jobs:"; cat $L/FAILED; }
echo "UMB_QUEUE_END"
