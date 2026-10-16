"""rcdl.inception_provenance: the protobuf reader, and the vendored InceptionV1's LRN against TensorFlow's definition."""

import numpy as np
import torch
import torch.nn.functional as F

from rcdl.inception_provenance import _fields, _tensor


def _varint(n):
  out = bytearray()
  while True:
    b = n & 0x7F
    n >>= 7
    out.append(b | (0x80 if n else 0))
    if not n:
      return bytes(out)


def _field(num, wt, payload):
  key = _varint((num << 3) | wt)
  return key + (_varint(len(payload)) + payload if wt == 2 else payload)


def test_tensorproto_reader():
  arr = np.arange(6, dtype='<f4').reshape(2, 3)
  shape = _field(2, 2, _field(1, 0, _varint(2))) + _field(2, 2, _field(1, 0, _varint(3)))
  msg = _field(1, 0, _varint(1)) + _field(2, 2, shape) + _field(4, 2, arr.tobytes())
  assert [n for n, _, _ in _fields(msg)] == [1, 2, 4]
  assert np.array_equal(_tensor(msg), arr)


def test_lrn_matches_tensorflow_definition():
  """TF LRN: x / (bias + alpha * sum_{|c'-c| <= r} x_c'^2)^beta; torch divides alpha by the window size."""
  x = torch.randn(2, 16, 5, 5, dtype=torch.float64)
  r, alpha, beta, bias = 5, 1e-4, 0.5, 2.0
  sq = F.pad(x.pow(2), (0, 0, 0, 0, r, r))
  s = torch.stack([sq[:, c:c + 2 * r + 1].sum(1) for c in range(16)], 1)
  ref = x / (bias + alpha * s) ** beta
  out = F.local_response_norm(x, size=2 * r + 1, alpha=alpha * (2 * r + 1), beta=beta, k=bias)
  assert torch.allclose(out, ref, atol=1e-12)
