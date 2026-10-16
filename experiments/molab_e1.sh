#!/usr/bin/env bash
# E1 on a fresh molab sandbox, resumable: ImageNet validation set and packed training set, then the trainer, which
# resumes from results/<run>/resume.pt when that exists (restored from the Hugging Face results dataset if present).
#
# Needs: /marimo/work/.hf_token (written separately, never in this repo), the repo at /marimo/work/repo.
# Usage: bash experiments/molab_e1.sh <run name> [trainer args...]
#   e.g. bash experiments/molab_e1.sh e1_tiny_adamw --size tiny --opt adamw --track all
# Logs: /marimo/work/e1_<run>.log (setup and training), /marimo/work/prep_train.log (packing).
set -euo pipefail
RUN=$1; shift
W=/marimo/work
REPO=joshhf/rcdl-molab-results            # private results dataset (the uploader writes results/ to neuron-dev/)
export HF_TOKEN_PATH=$W/.hf_token
cd $W/repo
mkdir -p $W/results/$RUN $W/data/imagenet

python3 - <<EOF
import os
from huggingface_hub import hf_hub_download, snapshot_download
tok = open('$W/.hf_token').read().strip()
if not os.path.exists('$W/data/imagenet/classes.py'):
  snapshot_download('ILSVRC/imagenet-1k', repo_type='dataset', token=tok, local_dir='$W/data/imagenet',
                    allow_patterns=['classes.py', 'data/validation-*'])
  print('validation set downloaded', flush=True)
for f in ('resume.pt', 'config.json', 'log.json', 'attr_init.pt'):
  if not os.path.exists('$W/results/$RUN/' + f):
    try:
      p = hf_hub_download('$REPO', f'neuron-dev/$RUN/{f}', repo_type='dataset', token=tok, local_dir='$W/hf_restore')
      os.replace(p, '$W/results/$RUN/' + f)
      print('restored', f, flush=True)
    except Exception as e:
      print('nothing to restore for', f, type(e).__name__, flush=True)
EOF

if [ ! -f /dev/shm/imagenet256/DONE ]; then
  python3 -c "from rcdl.imagenet import prepare_train; prepare_train(token=open('$W/.hf_token').read().strip())" \
    > $W/prep_train.log 2>&1
fi
cat /dev/shm/imagenet256/DONE; echo
python3 -m experiments.convnext_train --out $W/results/$RUN --val $W/data/imagenet/data "$@"
