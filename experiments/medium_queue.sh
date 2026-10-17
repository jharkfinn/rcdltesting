#!/usr/bin/env bash
# Stage 1 medium runs (medium plan, Sec. 5 1b) in order on one molab GPU, each through experiments/molab_medium.sh
# (local resume only, hf_sync without resume.pt, the stage3.block2 microscope after training), each logging to
# $W/<run>.log:
#   M10-s0  m10_medium_s0          medium,         stage3.block2 parts at 2,500 / 10,000 / 17,500 / 25,020
#   C13-s0  c13_medium_crush13_s0  medium_crush13, the crushed 13-unit stage3.block2, parts at every checkpoint
#   N8-s0   n8_medium_w8_s0        medium_w8,      parts as M10
#   N4-s0   n4_medium_w4_s0        medium_w4,      parts as M10
# Common: seed 0, the E1 recipe spelled out (AdamW 2e-3, weight decay 0.05, 1,250 warm-up steps, 25,020 steps, batch
# 512, mixup 0.8), --track stage3.block2, --snapshots --batch_crops, resume.pt every 2,500 steps and at the end only
# (--resume_at_ckpts 0). No --rand_groups (storage).
# Stability gate: while M10-s0 trains, its log.json is read every minute; if its top-1 at step 5,000 (the 10k-image
# eval) is below 0.10, M10-s0 is stopped and the queue prints GATE_FAILED and exits 1. A non-finite loss in M10-s0
# (trainer exit 3) also prints GATE_FAILED. Relaunch every run at a lower rate under new names:
#   EXTRA="--lr 1e-3" SUFFIX=_lr1e3 bash experiments/medium_queue.sh
# Any other failure of M10-s0 stops the queue (rerun it: finished runs are skipped, unfinished ones resume from their
# local resume.pt); a failure of a later run is reported and the queue goes on with the next. A failed microscope
# (molab_medium.sh exit 4: the run is trained) is reported and never stops the queue.
# Each run gets its own process group; stopping the queue (TERM, INT, HUP) stops the current run with it, and
# molab_medium.sh refuses a second instance on a run directory (exit 5), e.g. one left by a queue killed with KILL.
# Ends with MEDIUM_QUEUE_DONE, or MEDIUM_QUEUE_FAILED <runs> (<run>:microscope for a failed microscope only).
# Usage (on the sandbox): nohup bash experiments/medium_queue.sh > /marimo/work/medium_queue.log 2>&1 &
# env: W (/marimo/work), SUFFIX (run-name suffix), EXTRA (trainer args appended to every run), and those of
#      molab_medium.sh (HF_SYNC_RESTART, RESTART_SAME_NAME, MICROSCOPE, TRAIN, BLOCK)
set -uo pipefail +m                       # (no job control: setsid in launch then execs, so PID is the group)
W=${W:-/marimo/work}
SUFFIX=${SUFFIX:-}
EXTRA=${EXTRA:-}
GATE_STEP=5000
GATE_MIN=0.10
GATE_POLL=${GATE_POLL:-60}                               # seconds between reads of M10-s0's log.json
COMMON="--seed 0 --opt adamw --lr 2e-3 --wd 0.05 --warmup 1250 --steps 25020 --batch 512 --mixup 0.8
  --track stage3.block2 --snapshots --batch_crops --resume_every 2500 --resume_at_ckpts 0"
ATTR="--attr_steps 2500,10000,17500,25020"
cd $W/repo
echo "$(date +%T) medium queue on $(git log --oneline -1 2>/dev/null)"

PID=
trap '[ -n "$PID" ] && kill -TERM -- -$PID 2>/dev/null; echo "$(date +%T) queue stopped"; exit 143' TERM INT HUP

# launch <run> <args...>: molab_medium.sh in its own process group (PID = its group), in the background so that a
# signal to the queue is handled at once (wait returns)
launch() {
  local run=$1; shift
  echo "$(date +%T) start $run (log: $W/$run.log)"
  setsid bash experiments/molab_medium.sh $run "$@" > $W/$run.log 2>&1 &
  PID=$!
}

# gate <run>: 'pass', 'fail <top-1>' or 'pending' (no record at GATE_STEP yet) from the run's log.json
gate() {
  python3 - $W/results/$1/log.json $GATE_STEP $GATE_MIN <<'EOF'
import json, sys
try:
  log = json.load(open(sys.argv[1]))
except (OSError, ValueError):                           # not there yet, or being written
  log = []
r = [r['val_acc'] for r in log if r['step'] == int(sys.argv[2]) and r.get('val_acc') is not None]
print('pending' if not r else f'fail {r[-1]:.4f}' if r[-1] < float(sys.argv[3]) else 'pass')
EOF
}

# ---------------------------------------------------------------------------------- M10-s0, with the gate
FAILED=()
M10=m10_medium_s0$SUFFIX
launch $M10 $COMMON --size medium $ATTR $EXTRA
while kill -0 $PID 2>/dev/null; do
  G=$(gate $M10)
  if [[ $G == fail* ]]; then
    kill -TERM -- -$PID 2>/dev/null || kill -TERM $PID
    pkill -TERM -f "convnext_train --out $W/results/$M10 " 2>/dev/null
    wait $PID; PID=
    echo "$(date +%T) GATE_FAILED $M10: top-1 ${G#fail } at step $GATE_STEP is below $GATE_MIN; stopped. Relaunch" \
         "every run with EXTRA=\"--lr 1e-3\" SUFFIX=_lr1e3"
    exit 1
  fi
  sleep $GATE_POLL & wait $!
done
wait $PID
RC=$?; PID=
if [ $RC -eq 3 ]; then
  echo "$(date +%T) GATE_FAILED $M10: non-finite loss (see $W/$M10.log). Relaunch every run with" \
       "EXTRA=\"--lr 1e-3\" SUFFIX=_lr1e3"
  exit 1
elif [ $RC -eq 4 ]; then
  echo "$(date +%T) $M10 trained; its microscope failed (see $W/$M10.log)"
  FAILED+=($M10:microscope)
elif [ $RC -ne 0 ]; then
  echo "$(date +%T) $M10 failed (exit $RC: $(tail -1 $W/$M10.log)): queue stopped; rerun it to resume"
  exit 1
fi
G=$(gate $M10)
if [ "$G" != pass ]; then
  echo "$(date +%T) GATE_FAILED $M10: top-1 at step $GATE_STEP: ${G#fail } (below $GATE_MIN, or not logged)"
  exit 1
fi
echo "$(date +%T) $(tail -1 $W/$M10.log)"

# ------------------------------------------------------------------------------------- the other runs
one() {          # one <run> <trainer args...>
  local run=$1 rc; shift
  launch $run $COMMON "$@" $EXTRA
  wait $PID
  rc=$?; PID=
  if [ $rc -eq 0 ]; then
    echo "$(date +%T) $(tail -1 $W/$run.log)"
  elif [ $rc -eq 4 ]; then
    echo "$(date +%T) $run trained; its microscope failed (see $W/$run.log)"
    FAILED+=($run:microscope)
  else
    echo "$(date +%T) $run failed (exit $rc$([ $rc = 3 ] && echo ', non-finite loss'): $(tail -1 $W/$run.log))"
    FAILED+=($run)
  fi
}
one c13_medium_crush13_s0$SUFFIX --size medium_crush13
one n8_medium_w8_s0$SUFFIX --size medium_w8 $ATTR
one n4_medium_w4_s0$SUFFIX --size medium_w4 $ATTR
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "$(date +%T) MEDIUM_QUEUE_FAILED ${FAILED[*]}"
  exit 1
fi
echo "$(date +%T) MEDIUM_QUEUE_DONE"
