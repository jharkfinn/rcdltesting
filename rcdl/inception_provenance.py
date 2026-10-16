"""Provenance check for InceptionV1 (GoogLeNet), the model whose neurons the Distill Circuits work and Cammarata's
"Neural Networks Are Already Clean" name (mixed4e:55, mixed4c:447, the mixed3b curve detectors, ...).

Reference weights: Google's `tensorflow_inception_graph.pb` in download.tensorflow.org/models/inception5h.zip (the graph
Lucid loads as InceptionV1). PyTorch weights used here: `inception5h.pth` from ProGamerGov/pytorch-old-tensorflow-models,
which Lucent (Lucid for PyTorch) loads as `inceptionv1`. `compare(pb, pth)` reads every float constant of the TF graph
with a minimal protobuf wire-format reader (no TensorFlow needed) and compares it with the PyTorch tensor of the same
layer: conv kernels HWIO -> OIHW, dense kernels (in, out) -> (out, in).

Usage: python3 -m rcdl.inception_provenance <inception5h.zip or .pb> <inception5h.pth>
"""

import sys
import zipfile
from typing import Dict, Iterator, Tuple

import numpy as np
import torch


def _varint(buf: bytes, i: int) -> Tuple[int, int]:
  out, shift = 0, 0
  while True:
    b = buf[i]
    i += 1
    out |= (b & 0x7F) << shift
    if b < 0x80:
      return out, i
    shift += 7


def _fields(buf: bytes) -> Iterator[Tuple[int, int, object]]:
  """(field number, wire type, value) of a protobuf message; length-delimited values are returned as bytes."""
  i, n = 0, len(buf)
  while i < n:
    key, i = _varint(buf, i)
    num, wt = key >> 3, key & 7
    if wt == 0:
      v, i = _varint(buf, i)
    elif wt == 1:
      v, i = buf[i:i + 8], i + 8
    elif wt == 2:
      ln, i = _varint(buf, i)
      v, i = buf[i:i + ln], i + ln
    elif wt == 5:
      v, i = buf[i:i + 4], i + 4
    else:
      raise ValueError(f'unsupported wire type {wt}')
    yield num, wt, v


def _tensor(buf: bytes) -> np.ndarray:
  """TensorProto (float32 only): 1 dtype, 2 tensor_shape (2 dim: 1 size), 4 tensor_content, 5 float_val."""
  dtype, shape, content, floats = None, [], None, []
  for num, wt, v in _fields(buf):
    if num == 1:
      dtype = v
    elif num == 2:
      for dn, _, dv in _fields(v):
        if dn == 2:
          shape.append(next(sv for sn, _, sv in _fields(dv) if sn == 1) if dv else 0)
    elif num == 4:
      content = v
    elif num == 5:
      floats.extend(np.frombuffer(v, '<f4').tolist() if wt == 2 else [np.frombuffer(v, '<f4')[0]])
  if dtype != 1:                                          # DT_FLOAT
    return None
  arr = np.frombuffer(content, '<f4') if content is not None else np.array(floats, '<f4')
  if arr.size == 1 and int(np.prod(shape)) > 1:
    arr = np.full(int(np.prod(shape)), arr[0], '<f4')
  return arr.reshape(shape) if shape else arr


def graph_constants(pb: bytes) -> Dict[str, np.ndarray]:
  """{node name: float32 array} for every Const node of a serialized GraphDef."""
  out = {}
  for num, _, node in _fields(pb):
    if num != 1:
      continue
    name = op = None
    value = None
    for n2, _, v in _fields(node):
      if n2 == 1:
        name = v.decode()
      elif n2 == 2:
        op = v.decode()
      elif n2 == 5:                                       # attr map entry: 1 key, 2 AttrValue (8 tensor)
        key = attr = None
        for n3, _, v3 in _fields(v):
          if n3 == 1:
            key = v3.decode()
          elif n3 == 2:
            attr = v3
        if key == 'value' and attr is not None:
          for n4, _, v4 in _fields(attr):
            if n4 == 8:
              value = _tensor(v4)
    if op == 'Const' and value is not None:
      out[name] = value
  return out


def compare(pb_path: str, pth_path: str) -> Dict[str, object]:
  if pb_path.endswith('.zip'):
    pb = zipfile.ZipFile(pb_path).read('tensorflow_inception_graph.pb')
  else:
    pb = open(pb_path, 'rb').read()
  tf = graph_constants(pb)
  sd = torch.load(pth_path, map_location='cpu', weights_only=True)
  matched, worst, missing = 0, 0.0, []
  for key, t in sd.items():
    base, kind = key.rsplit('.', 1)
    base = base.replace('_pre_relu_conv', '').replace('_pre_activation_matmul', '')   # Lucent names -> TF names
    ref = tf.get(f"{base}_{'w' if kind == 'weight' else 'b'}")
    if ref is None:
      missing.append(key)
      continue
    if kind == 'weight':
      ref = ref.transpose(3, 2, 0, 1) if ref.ndim == 4 else ref.T
    if tuple(ref.shape) != tuple(t.shape):
      missing.append(f'{key}: shape {tuple(t.shape)} vs {tuple(ref.shape)}')
      continue
    worst = max(worst, float(np.abs(ref - t.numpy()).max()))
    matched += 1
  used = {k.rsplit('.', 1)[0].replace('_pre_relu_conv', '').replace('_pre_activation_matmul', '') for k in sd}
  not_ported = sorted({n.rsplit('_', 1)[0] for n in tf} - used)     # auxiliary heads of GoogLeNet training
  return {'pth_tensors': len(sd), 'matched': matched, 'max_abs_diff': worst, 'unmatched': missing,
          'tf_consts': len(tf), 'tf_layers_not_in_pth': not_ported}


if __name__ == '__main__':
  print(compare(sys.argv[1], sys.argv[2]))
