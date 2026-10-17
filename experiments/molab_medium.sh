#!/usr/bin/env bash
# A medium ConvNeXt run on a molab sandbox (medium plan, Sec. 5), then its microscope:
#   1. validation set and packed training set: reused when present, else prepared as experiments/molab_e1.sh does
#      (/dev/shm/imagenet256 takes 22-24 min to pack);
#   2. resume states stay local: a medium resume.pt (about 13 GB with stage3.block2 tracked) is never restored from or
#      uploaded to Hugging Face. Before training, this script asks which running hf_sync processes would upload
#      results/<run>/resume.pt (experiments/hf_sync.py sync_status, from their command lines):
#        none   -> it starts one for $W/results with --extra_ignore '*/resume.pt' (results upload as before);
#        one that ignores it -> reused;
#        one that would upload it -> stop with exit 1 (HF_SYNC_UPLOADS_RESUME), or with HF_SYNC_RESTART=1 replace it
#        by the same command plus --extra_ignore '*/resume.pt'.
#      Checked here rather than left to the caller: an hf_sync already running for these results from earlier runs
#      (molab_e1.sh era, without the flag) would otherwise upload 13 GB every 10 minutes unnoticed;
#   3. the trainer, which resumes from results/<run>/resume.pt only if that exists locally (a new sandbox starts the
#      run again from step 0); its exit code is passed on (3: non-finite loss);
#   4. experiments/convnext_microscope on the final model for $BLOCK (stage3.block2) with --dirs min(32, units of the
#      block) (32; 13 for the crushed 13-unit block), into results/<run>/microscope;
#   5. prints <RUN>_DONE (the run name in capitals, e.g. M10_MEDIUM_S0_DONE).
# A trained run (its final model and log record exist) is not trained again and a finished microscope is not redone,
# so a rerun continues where the previous one stopped; a run whose resume.pt was pruned is never restarted from 0.
#
# Needs: /marimo/work/.hf_token (never printed), the repo at /marimo/work/repo.
# Usage: bash experiments/molab_medium.sh <run> [trainer args...]
#   e.g. bash experiments/molab_medium.sh m10_medium_s0 --size medium --seed 0 --track stage3.block2 \
#          --attr_steps 2500,10000,17500,25020 --resume_every 2500 --resume_at_ckpts 0 --snapshots --batch_crops
# env: W (/marimo/work), TRAIN (/dev/shm/imagenet256), BLOCK (stage3.block2), HF_SYNC_RESTART (0),
#      HF_SYNC (auto; off = no check and no start, for local tests only)
set -euo pipefail
RUN=$1; shift
W=${W:-/marimo/work}
TRAIN=${TRAIN:-/dev/shm/imagenet256}
BLOCK=${BLOCK:-stage3.block2}
HF_SYNC=${HF_SYNC:-auto}
HF_SYNC_RESTART=${HF_SYNC_RESTART:-0}
ROOT=$W/results
OUT=$ROOT/$RUN
VAL=$W/data/imagenet/data
TAG=${RUN^^}
export HF_TOKEN_PATH=$W/.hf_token PYTHONUNBUFFERED=1
cd $W/repo
mkdir -p $OUT $W/data/imagenet
echo "$(date +%T) $RUN on $(git log --oneline -1 2>/dev/null || echo 'no git')"

# ------------------------------------------------------------------------------ 1 validation and training sets
if ! compgen -G "$VAL/validation-*.parquet" > /dev/null; then          # molab may restore classes.py alone
  python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download('ILSVRC/imagenet-1k', repo_type='dataset', token=open('$W/.hf_token').read().strip(),
                  local_dir='$W/data/imagenet', allow_patterns=['classes.py', 'data/validation-*'])
print('validation set downloaded', flush=True)
EOF
fi
if [ ! -f $TRAIN/DONE ]; then
  echo "$(date +%T) packing the training set into $TRAIN (log: $W/prep_train.log)"
  python3 -c "from rcdl.imagenet import prepare_train; prepare_train('$TRAIN', token=open('$W/.hf_token').read().strip())" \
    > $W/prep_train.log 2>&1
fi
cat $TRAIN/DONE; echo

# ------------------------------------------------------------------------------- 2 hf_sync without resume.pt
if [ "$HF_SYNC" != off ]; then
  SYNC=$(python3 -c "import sys; from experiments.hf_sync import sync_status; print(sync_status(sys.argv[1]))" \
    $OUT/resume.pt)
  if [[ $SYNC == uploaded:* ]] && [ "$HF_SYNC_RESTART" = 1 ]; then
    python3 - ${SYNC#uploaded:} $W/hf_sync.log <<'EOF'
import os, signal, subprocess, sys, time
from experiments.hf_sync import running_syncs
pids = {int(p) for p in sys.argv[1].split(',')}
for pid, argv, _ in running_syncs():
  if pid not in pids:
    continue
  cwd = os.readlink(f'/proc/{pid}/cwd')
  os.kill(pid, signal.SIGTERM)
  for _ in range(50):
    if not os.path.exists(f'/proc/{pid}'):
      break
    time.sleep(0.2)
  subprocess.Popen(argv + ['--extra_ignore', '*/resume.pt'], cwd=cwd, start_new_session=True,
                   stdout=open(sys.argv[2], 'a'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
  print('hf_sync', pid, 'replaced with --extra_ignore */resume.pt', flush=True)
EOF
    sleep 2
    SYNC=$(python3 -c "import sys; from experiments.hf_sync import sync_status; print(sync_status(sys.argv[1]))" \
      $OUT/resume.pt)
  fi
  case $SYNC in
    none)
      setsid nohup python3 experiments/hf_sync.py --root $ROOT --prefix neuron-dev --every 600 \
        --extra_ignore '*/resume.pt' >> $W/hf_sync.log 2>&1 < /dev/null &
      disown
      echo "hf_sync started for $ROOT without resume.pt (log: $W/hf_sync.log)";;
    ignored)
      echo "hf_sync running for $ROOT leaves resume.pt out";;
    *)
      echo "HF_SYNC_UPLOADS_RESUME: hf_sync process(es) ${SYNC#uploaded:} would upload $OUT/resume.pt. Restart them" \
           "with --extra_ignore '*/resume.pt' (or rerun with HF_SYNC_RESTART=1); not training."
      exit 1;;
  esac
fi

# ------------------------------------------------------------------------------------------------ 3 training
# last_step: the run's last step from its config.json (--max_steps if it stops early), empty if there is none
last_step() {
  python3 -c "import json,sys; c=json.load(open(sys.argv[1])); m=c.get('max_steps') or 0;\
print(m if 0 < m < c['steps'] else c['steps'])" $OUT/config.json 2>/dev/null || true
}
trained() {      # the final model and the final log record exist
  local s; s=$(last_step)
  [ -n "$s" ] && [ -f $OUT/model_$(printf %05d $s).pt ] && python3 -c "import json,sys;\
sys.exit(0 if any(r['step'] == int(sys.argv[2]) for r in json.load(open(sys.argv[1]))) else 1)" $OUT/log.json $s \
    2>/dev/null
}
if trained; then
  echo "$(date +%T) $RUN already trained (step $(last_step)); not training again"
else
  [ -f $OUT/resume.pt ] && echo "$(date +%T) resuming from the local resume.pt" || echo "$(date +%T) starting from step 0"
  RC=0
  python3 -m experiments.convnext_train --out $OUT --val $VAL --train $TRAIN "$@" || RC=$?
  if [ $RC -ne 0 ]; then
    echo "${TAG}_FAILED: trainer exit $RC$([ $RC = 3 ] && echo ' (non-finite loss)')"
    exit $RC
  fi
  trained || { echo "${TAG}_FAILED: no final model or log record after training"; exit 1; }
fi

# ---------------------------------------------------------------------------------------------- 4 microscope
LAST=$(last_step)
LAYER=${BLOCK/./_}
if [ -f $OUT/microscope/data_$LAYER.json ]; then
  echo "microscope of $BLOCK already written"
else
  U=$(python3 -c "import json,sys; from rcdl.convnext import build_convnext;\
print(dict(build_convnext(json.load(open(sys.argv[1]))['size']).blocks())[sys.argv[2]].pw_down.in_features)" \
    $OUT/config.json $BLOCK)
  D=$(( U < 32 ? U : 32 ))
  echo "$(date +%T) microscope: $BLOCK ($U units) at step $LAST with $D random directions"
  python3 -m experiments.convnext_microscope --run $OUT --step $LAST --block $BLOCK --dirs $D --val $VAL \
    --imagenet $W/data/imagenet --out $OUT/microscope
fi
echo "$(date +%T) ${TAG}_DONE"
