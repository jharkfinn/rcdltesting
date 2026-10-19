"""Checkpoints stored as chunked base64 text in .py files.

molab restores only the .py and .md files of a notebook's work directory when it replaces a reclaimed sandbox, so a
checkpoint that must survive that is written as `<prefix>_s<step>_p<i>.py` parts (`chunk_mb` MB of base64 each) and a
`<prefix>_meta.py` file that names the parts and their sha256. The meta file is replaced last (atomic rename), and the
parts of the previous checkpoint are deleted only after it, so an interrupted save leaves the previous checkpoint
readable. `load` returns None when there is no checkpoint or when a part is missing or corrupt.
"""

import base64
import glob
import hashlib
import io
import json
import os
from typing import Any, Optional

import torch


def _write(path: str, text: str):
  tmp = path + '.tmp'
  with open(tmp, 'w') as f:
    f.write(text)
  os.replace(tmp, path)


def save(prefix: str, obj: Any, step: int, chunk_mb: float = 8.0):
  buf = io.BytesIO()
  torch.save(obj, buf)
  raw = buf.getvalue()
  b64 = base64.b64encode(raw).decode()
  n = max(1, int(chunk_mb * 2 ** 20))
  parts = []
  for i in range(0, len(b64), n):
    path = f'{prefix}_s{step}_p{i // n:03d}.py'
    _write(path, f"DATA = '{b64[i:i + n]}'\n")
    parts.append(os.path.basename(path))
  meta = {'step': step, 'parts': parts, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
  _write(f'{prefix}_meta.py', f'META = {json.dumps(meta)!r}\n')
  keep = set(parts)
  for old in glob.glob(f'{glob.escape(prefix)}_s*_p*.py'):
    if os.path.basename(old) not in keep:
      os.remove(old)


def load(prefix: str, map_location=None) -> Optional[Any]:
  try:
    text = open(f'{prefix}_meta.py').read()
    meta = json.loads(text[len('META = '):].strip()[1:-1])
    d = os.path.dirname(prefix)
    b64 = ''.join(open(os.path.join(d, p)).read()[len("DATA = '"):].rstrip()[:-1] for p in meta['parts'])
    raw = base64.b64decode(b64)
  except (OSError, ValueError, KeyError):
    return None
  if hashlib.sha256(raw).hexdigest() != meta['sha256']:
    return None
  return torch.load(io.BytesIO(raw), map_location=map_location, weights_only=False)


def remove(prefix: str):
  for f in glob.glob(f'{glob.escape(prefix)}_s*_p*.py') + glob.glob(f'{glob.escape(prefix)}_meta.py'):
    os.remove(f)
