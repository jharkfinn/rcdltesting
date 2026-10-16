"""Copies a results folder to a private Hugging Face dataset every few minutes (sandboxes can end without notice).

Usage: python3 experiments/hf_sync.py [--root /marimo/work/results] [--prefix neuron-dev] [--every 600]
The token is read from /marimo/work/.hf_token (or --token_file); it is never printed. Unchanged files are not
uploaded again (the Hub skips content it already has). Partial files (*.tmp) are skipped.
"""

import argparse
import os
import time

from huggingface_hub import HfApi


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--root', default='/marimo/work/results')
  p.add_argument('--prefix', default='neuron-dev')
  p.add_argument('--repo', default='joshhf/rcdl-molab-results')
  p.add_argument('--token_file', default='/marimo/work/.hf_token')
  p.add_argument('--every', type=int, default=600)
  a = p.parse_args()
  api = HfApi(token=open(a.token_file).read().strip())
  api.create_repo(a.repo, repo_type='dataset', private=True, exist_ok=True)
  while True:
    try:
      if any(files for _, _, files in os.walk(a.root)):
        api.upload_folder(folder_path=a.root, path_in_repo=a.prefix, repo_type='dataset', repo_id=a.repo,
                          ignore_patterns=['*.tmp', 'bench*', '*buffers*'],
                          commit_message=f'{a.prefix}: results at {time.strftime("%H:%M UTC", time.gmtime())}')
        print(time.strftime('%H:%M:%S'), 'uploaded', flush=True)
    except Exception as e:                       # keep going; report the error type only
      print(time.strftime('%H:%M:%S'), 'upload failed:', type(e).__name__, str(e)[:200], flush=True)
    time.sleep(a.every)


if __name__ == '__main__':
  main()
