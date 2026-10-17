#!/usr/bin/env bash
# A medium ConvNeXt run on a molab sandbox (medium plan, Sec. 5), then its microscope:
#   0. one instance per run directory: a lock (in $LOCKS, outside the uploaded results) that the trainer inherits, so
#      it stays held while a trainer orphaned by a killed queue runs on; a second instance exits 5;
#   1. validation set and packed training set: reused when present, else prepared as experiments/molab_e1.sh does
#      (/dev/shm/imagenet256 takes 22-24 min to pack);
#   2. resume states stay local: a medium resume.pt (about 6.4 GB with stage3.block2 tracked, 13 GB with --rand_groups)
#      is never restored from or uploaded to Hugging Face. experiments/hf_sync.py ensure_sync (under a lock, so runs
#      started at once start one sync) asks which running hf_sync processes would upload results/<run>/resume.pt:
#        none   -> it starts one for $W/results with --extra_ignore '*/resume.pt' (results upload as before);
#        one that ignores it -> reused;
#        one that would upload it -> exit 1 (HF_SYNC_UPLOADS_RESUME), or with HF_SYNC_RESTART=1 it is stopped and
#          started again from $W/repo with its own arguments plus --extra_ignore '*/resume.pt' (refused while non-medium
#          trainers write under its root);
#      a started sync must still run after HF_SYNC_WAIT s, else exit 7 (HF_SYNC_DEAD); the token file is checked
#      first. Checked here rather than left to the caller: an hf_sync already running for these results from earlier
#      runs (molab_e1.sh era, without the flag) would otherwise upload 6.4 GB every 10 minutes unnoticed.
#      The pattern covers EVERY run under $W/results, and the sync outlives this script: a molab_e1.sh run there loses
#      its HF resume state (molab_e1.sh warns; ensure_sync names such trainers);
#   3. the trainer, which resumes from results/<run>/resume.pt only if that exists locally. A run new to this sandbox
#      whose log.json is already on HF (an earlier attempt on a lost sandbox) exits 6 unless RESTART_SAME_NAME=1: the
#      restart from step 0 rewrites that HF folder file by file, which holds a mix of both attempts until it passes the
#      old attempt's last step (or use a new run name). The trainer's exit code is passed on (3: non-finite loss);
#   4. experiments/convnext_microscope on the final model for $BLOCK (stage3.block2) with --dirs min(32, units of the
#      block) (32; 13 for the crushed 13-unit block), into results/<run>/microscope. A failure there prints
#      <RUN>_MICROSCOPE_FAILED and exits 4 after <RUN>_DONE (the run is trained); MICROSCOPE=0 skips it;
#   5. the hf_sync check again (the last run's results must still upload; exit 7 if not), then <RUN>_DONE (the run
#      name in capitals, e.g. M10_MEDIUM_S0_DONE).
# A trained run (its final model and log record exist) is not trained again and a finished microscope is not redone,
# so a rerun continues where the previous one stopped; a run whose resume.pt was pruned is never restarted from 0. The
# final resume.pt stays (pruned by the analysis tail after a verified upload, plan Sec. 5 1c).
#
# Needs: /marimo/work/.hf_token (never printed), the repo at /marimo/work/repo.
# Usage: bash experiments/molab_medium.sh <run> [trainer args...]
#   e.g. bash experiments/molab_medium.sh m10_medium_s0 --size medium --seed 0 --track stage3.block2 \
#          --attr_steps 2500,10000,17500,25020 --resume_every 2500 --resume_at_ckpts 0 --snapshots --batch_crops
# Exit codes: 0 done, 1 setup or hf_sync refusal, 3 non-finite loss, 4 microscope failed, 5 already running, 6 earlier
#   attempt on HF, 7 hf_sync not running; others from the trainer.
# env: W (/marimo/work), TRAIN (/dev/shm/imagenet256), BLOCK (stage3.block2), HF_SYNC_RESTART (0), HF_SYNC_WAIT (15),
#      RESTART_SAME_NAME (0), MICROSCOPE (1), LOCKS (/tmp/rcdl_locks),
#      HF_SYNC (auto; off = no hf_sync and no HF lookup: local tests, or training without uploads)
set -euo pipefail
RUN=$1; shift
W=${W:-/marimo/work}
TRAIN=${TRAIN:-/dev/shm/imagenet256}
BLOCK=${BLOCK:-stage3.block2}
HF_SYNC=${HF_SYNC:-auto}
HF_SYNC_RESTART=${HF_SYNC_RESTART:-0}
HF_SYNC_WAIT=${HF_SYNC_WAIT:-15}
RESTART_SAME_NAME=${RESTART_SAME_NAME:-0}
MICROSCOPE=${MICROSCOPE:-1}
LOCKS=${LOCKS:-/tmp/rcdl_locks}
ROOT=$W/results
OUT=$ROOT/$RUN
VAL=$W/data/imagenet/data
TAG=${RUN^^}
export HF_TOKEN_PATH=$W/.hf_token PYTHONUNBUFFERED=1
cd $W/repo
mkdir -p $OUT $W/data/imagenet $LOCKS
echo "$(date +%T) $RUN on $(git log --oneline -1 2>/dev/null || echo 'no git')"

# ------------------------------------------------------------------------------- 0 one instance per run directory
LOCK=$LOCKS/run$(realpath $OUT | tr / _).lock
exec 9> $LOCK
if ! flock -n 9; then
  echo "${TAG}_FAILED: already running for $OUT (lock $LOCK; trainer pids:" \
       "$(pgrep -f "convnext_train --out $OUT " | tr '\n' ' '))"
  exit 5
fi

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
  python3 -c "from rcdl.imagenet import prepare_train;\
prepare_train('$TRAIN', token=open('$W/.hf_token').read().strip())" > $W/prep_train.log 2>&1
fi
cat $TRAIN/DONE; echo

# ------------------------------------------------------------------------------- 2 hf_sync without resume.pt
sync_ok() {      # an hf_sync leaves $OUT/resume.pt out (started or restarted if needed); 0, 1 or 7 (ensure_sync)
  python3 -c "import sys; from experiments.hf_sync import ensure_sync as e; v = sys.argv[1:];\
sys.exit(e(v[0], v[1], v[2], v[3], restart=v[4] == '1', token_file=v[5], lock=v[6], wait=float(v[7])))" \
    $OUT/resume.pt $ROOT $W/repo $W/hf_sync.log $HF_SYNC_RESTART $W/.hf_token $LOCKS/hf_sync.lock $HF_SYNC_WAIT
}
if [ "$HF_SYNC" != off ]; then
  RC=0; sync_ok || RC=$?
  if [ $RC -ne 0 ]; then
    echo "${TAG}_FAILED: no hf_sync that uploads the results without resume.pt (exit $RC, see above); not training"
    exit $RC
  fi
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
  if [ -f $OUT/resume.pt ]; then
    echo "$(date +%T) resuming from the local resume.pt"
  else
    if [ "$HF_SYNC" != off ] && [ ! -f $OUT/config.json ] && [ "$RESTART_SAME_NAME" != 1 ]; then
      RC=0                                                # new to this sandbox: an earlier attempt on HF?
      python3 - $RUN $W/.hf_token <<'EOF' || RC=$?
import sys
from huggingface_hub import HfApi
from experiments.hf_sync import parser
d = parser().parse_args([])
f = f'{d.prefix}/{sys.argv[1]}/log.json'
try:
  there = HfApi(token=open(sys.argv[2]).read().strip()).file_exists(d.repo, f, repo_type='dataset')
except Exception as e:                                    # report the error type only
  print('WARNING: could not look for an earlier attempt on HF:', type(e).__name__, flush=True)
  there = False
if there:
  print(f'{d.repo}: {f} exists (an earlier attempt of this run, on another sandbox)', flush=True)
  sys.exit(10)
EOF
      if [ $RC = 10 ]; then
        echo "${TAG}_FAILED: an earlier attempt is on HF; starting again from step 0 would mix its files with this" \
             "one's until it passes the old last step. RESTART_SAME_NAME=1 to do so, or use a new run name"
        exit 6
      fi
    fi
    echo "$(date +%T) starting from step 0"
  fi
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
microscope() {   # (set -e does not apply in here: every step checked)
  local u d
  u=$(python3 -c "import json,sys; from rcdl.convnext import build_convnext;\
print(dict(build_convnext(json.load(open(sys.argv[1]))['size']).blocks())[sys.argv[2]].pw_down.in_features)" \
    $OUT/config.json $BLOCK) || return 1
  d=$(( u < 32 ? u : 32 ))
  echo "$(date +%T) microscope: $BLOCK ($u units) at step $LAST with $d random directions"
  python3 -m experiments.convnext_microscope --run $OUT --step $LAST --block $BLOCK --dirs $d --val $VAL \
    --imagenet $W/data/imagenet --out $OUT/microscope || return $?
}
NOTE=; END=0
if [ "$MICROSCOPE" = 0 ]; then
  echo "microscope skipped (MICROSCOPE=0)"
elif [ -f $OUT/microscope/data_$LAYER.json ]; then
  echo "microscope of $BLOCK already written"
else
  RC=0; microscope || RC=$?
  if [ $RC -ne 0 ]; then
    echo "$(date +%T) ${TAG}_MICROSCOPE_FAILED: exit $RC (the run is trained; rerun this script for the microscope," \
         "or MICROSCOPE=0)"
    NOTE=" (microscope failed)"; END=4
  fi
fi

# ------------------------------------------------------------------------------- 5 the results still upload
if [ "$HF_SYNC" != off ]; then
  RC=0; sync_ok || RC=$?
  if [ $RC -ne 0 ]; then
    echo "HF_SYNC_DEAD: no hf_sync uploads the results of $RUN now (exit $RC, see above)"
    NOTE="$NOTE (hf_sync not running)"; END=7
  fi
fi
echo "$(date +%T) ${TAG}_DONE$NOTE"
exit $END
