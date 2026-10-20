"""Paper App. E.3.1 (claim 18): SGD with DeltaNet-like decay on a 10-block residual MLP, CIFAR-10, 2,000 epochs.

Update of every linear layer (paper, App. E.3.1):  W_t = W_{t-1} (I - λ η_t X_t X_t^T) - η_t G_t X_t^T,
  X_t  (d_in x B):  the layer's stacked inputs in the batch,
  G_t  (d_out x B): the stacked gradients of the batch-mean loss w.r.t. the layer's outputs,
  η_t:  the learning rate at step t (5 epochs of linear warmup, then cosine annealing to 0),
  λ:    the Delta-rule decay rate (λ = 0 is plain SGD).
Since W X = Y0 (the layer output without bias), the update is an SGD step whose weight gradient is G X^T + λ Y0 X^T,
i.e. the gradient of the loss plus (λ/2) Σ_b ||W x_b||^2 with the inputs held fixed. Biases and LayerNorm parameters
get plain SGD.

Model (paper Table "DeltaNet-GD"): Linear 3072 -> 512, 10 pre-LayerNorm residual blocks x <- x + Dropout(W2 GELU(W1
LN(x))) with width 2048, final LayerNorm, Linear 512 -> 10. Batch 64, SGD lr 1e-2, momentum 0, dropout 0.1.

Efficiency: K models (one per λ) are trained in one process as stacked weights (batched matmuls), on the same images
in the same order; augmentation runs on the GPU once per epoch; one training step (forward, backward, Delta term,
SGD update) is captured as a CUDA graph and replayed. Assumptions where the paper is silent: PyTorch default Linear
init, exact GELU, dropout on the residual branch only, warmup from 0, the multi-epoch augmentation of App. E.1.1
(reflect-pad 4, random crop, random flip), per-channel standardization.

Usage: python -m experiments.deltanet_long --out results/deltanet --epochs 2000 --lams 0 7e-6 1e-5 2e-5 3e-5
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from rcdl.data import CIFAR_MEAN, CIFAR_STD, load_cifar10_uint8


class StackedResMLP:
  """K independent residual MLPs as stacked parameter tensors (leading dimension K)."""

  def __init__(self, K: int, d_in=3072, d=512, hidden=2048, blocks=10, n_cls=10, dropout=0.1, seed=0, device='cpu'):
    g = torch.Generator().manual_seed(seed)

    def linear(d_out, d_in_):                 # PyTorch nn.Linear default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
      bound = 1.0 / math.sqrt(d_in_)
      w = (torch.rand(K, d_out, d_in_, generator=g) * 2 - 1) * bound
      b = (torch.rand(K, d_out, generator=g) * 2 - 1) * bound
      return w.to(device).requires_grad_(), b.to(device).requires_grad_()

    ones = lambda: torch.ones(K, d, device=device, requires_grad=True)
    zeros = lambda: torch.zeros(K, d, device=device, requires_grad=True)
    self.K, self.p_drop, self.blocks = K, dropout, blocks
    self.W_in, self.b_in = linear(d, d_in)
    self.blk = [dict(zip(('W1', 'b1'), linear(hidden, d))) | dict(zip(('W2', 'b2'), linear(d, hidden)))
                | {'g': ones(), 'b': zeros()} for _ in range(blocks)]
    self.g_f, self.b_f = ones(), zeros()
    self.W_out, self.b_out = linear(n_cls, d)

  def params(self):
    ps = [self.W_in, self.b_in, self.g_f, self.b_f, self.W_out, self.b_out]
    for b in self.blk:
      ps += [b['W1'], b['b1'], b['W2'], b['b2'], b['g'], b['b']]
    return ps

  @staticmethod
  def _ln(x, g, b):
    return F.layer_norm(x, x.shape[-1:]) * g[:, None, :] + b[:, None, :]

  def forward(self, x: torch.Tensor, train: bool, c=None, record=None):
    """x: (B, d_in) shared by all K models. c: (K,) Delta coefficients λ_k η_t (training) or None (plain linear
    layers). record: list that receives (W, X, Y0) of every linear layer (tests)."""
    K, B = self.K, x.shape[0]
    if c is None:
      y0 = (x @ self.W_in.reshape(-1, x.shape[1]).T).reshape(B, K, -1).transpose(0, 1)   # (K, B, d)
      bmm = lambda a, W: torch.bmm(a, W.transpose(1, 2))
    else:
      y0 = DeltaLinearShared.apply(x, self.W_in, c)
      bmm = lambda a, W: DeltaLinear.apply(a, W, c)
    if record is not None:
      record.append((self.W_in, x, y0))
    h = y0 + self.b_in[:, None, :]
    for blk in self.blk:
      a = self._ln(h, blk['g'], blk['b'])
      u0 = bmm(a, blk['W1'])
      z = F.gelu(u0 + blk['b1'][:, None, :])
      v0 = bmm(z, blk['W2'])
      if record is not None:
        record += [(blk['W1'], a, u0), (blk['W2'], z, v0)]
      v = v0 + blk['b2'][:, None, :]
      h = h + (F.dropout(v, self.p_drop, training=True) if train else v)
    a = self._ln(h, self.g_f, self.b_f)
    o0 = bmm(a, self.W_out)
    if record is not None:
      record.append((self.W_out, a, o0))
    return o0 + self.b_out[:, None, :]                                                  # (K, B, n_cls)


class DeltaLinear(torch.autograd.Function):
  """Y0 = X W^T per model (X: (K, B, d_in), W: (K, d_out, d_in)). Backward: dX = G W as usual, but the weight
  gradient is (G + c Y0)^T X with c = λ η per model: the Delta-rule decay enters the weight update in the same
  matmul, and does not change the gradient passed to earlier layers."""

  @staticmethod
  def forward(ctx, x, W, c):
    y0 = torch.bmm(x, W.transpose(1, 2))
    ctx.save_for_backward(x, W, c, y0)
    return y0

  @staticmethod
  def backward(ctx, G):
    x, W, c, y0 = ctx.saved_tensors
    dx = torch.bmm(G, W) if ctx.needs_input_grad[0] else None
    dW = torch.bmm((G + c[:, None, None] * y0).transpose(1, 2), x)
    return dx, dW, None


class DeltaLinearShared(torch.autograd.Function):
  """DeltaLinear for the input layer, whose input X (B, d_in) is shared by all K models (one GEMM for all)."""

  @staticmethod
  def forward(ctx, x, W, c):
    K, d_out, d_in = W.shape
    y0 = (x @ W.reshape(-1, d_in).T).reshape(x.shape[0], K, d_out).transpose(0, 1)
    ctx.save_for_backward(x, c, y0)
    ctx.shape = W.shape
    return y0

  @staticmethod
  def backward(ctx, G):
    x, c, y0 = ctx.saved_tensors
    K, d_out, d_in = ctx.shape
    S = (G + c[:, None, None] * y0).transpose(1, 2).reshape(K * d_out, x.shape[0])
    return None, (S @ x).reshape(K, d_out, d_in), None


def train_step(model, params, x, y, lam, lr, stats=None):
  """One SGD step with Delta-rule decay for all K models. lr: 0-dim tensor (CUDA-graph safe). Returns the K losses.

  The loss is scaled by η_t before the backward pass, so every gradient already is the step: p <- p - grad. With the
  Delta coefficient c = λ η_t, the weight step is η_t (G + λ Y0)^T X = η_t G X^T + λ η_t W X X^T, the paper's update."""
  K, B = model.K, x.shape[0]
  out = model.forward(x, train=True, c=lam * lr)
  losses = F.cross_entropy(out.reshape(-1, out.shape[-1]), y.repeat(K), reduction='none').reshape(K, B).mean(1)
  (losses.sum() * lr).backward()                 # models are independent: d(sum)/dθ_k = d(loss_k)/dθ_k
  with torch.no_grad():
    torch._foreach_sub_(params, [q.grad for q in params])
    if stats is not None:
      stats[0].add_(losses)
      stats[1].add_((out.argmax(-1) == y).float().sum(1))
  return losses.detach()


def lr_at(step, total, warm, base):
  if step < warm:
    return base * (step + 1) / warm
  return base * 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def gpu_augment(padded: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
  """App. E.1.1 augmentation on the GPU: random 32x32 crop of the reflect-padded (n, 40, 40, 3) images, random flip."""
  n, dev = padded.shape[0], padded.device
  ox = torch.randint(0, 9, (n,), generator=gen, device=dev)
  oy = torch.randint(0, 9, (n,), generator=gen, device=dev)
  ar = torch.arange(32, device=dev)
  rows, cols = oy[:, None] + ar, ox[:, None] + ar
  out = padded[torch.arange(n, device=dev)[:, None, None], rows[:, :, None], cols[:, None, :]]
  flip = torch.rand(n, generator=gen, device=dev) < 0.5
  return torch.where(flip[:, None, None, None], out.flip(2), out)


def standardize(u8: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
  return ((u8.float() / 255.0 - mean) / std).reshape(u8.shape[0], -1)


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
  correct = torch.zeros(model.K, device=x.device)
  loss = torch.zeros(model.K, device=x.device)
  for i in range(0, len(x), bs):
    out = model.forward(x[i:i + bs], train=False)
    correct += (out.argmax(-1) == y[i:i + bs]).float().sum(1)
    loss += F.cross_entropy(out.reshape(-1, out.shape[-1]), y[i:i + bs].repeat(model.K), reduction='none'
                            ).reshape(model.K, -1).sum(1)
  return (loss / len(x)).tolist(), (correct / len(x)).tolist()


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out', default='results/deltanet')
  p.add_argument('--epochs', type=int, default=2000)
  p.add_argument('--lams', type=float, nargs='+', default=[0.0, 7e-6, 1e-5, 2e-5, 3e-5])
  p.add_argument('--lr', type=float, default=1e-2)
  p.add_argument('--batch', type=int, default=64)
  p.add_argument('--warmup_epochs', type=int, default=5)
  p.add_argument('--eval_every', type=int, default=10, help='epochs')
  p.add_argument('--train_eval', type=int, default=10000, help='non-augmented training images per evaluation')
  p.add_argument('--seed', type=int, default=1)
  p.add_argument('--no_graph', action='store_true', help='eager steps (CPU tests)')
  p.add_argument('--max_steps', type=int, default=None, help='speed tests')
  p.add_argument('--data', default='data')
  p.add_argument('--source', default='auto', help="CIFAR source of rcdl.data: 'auto' | 'binary' | 'parquet'")
  args = p.parse_args()

  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.manual_seed(args.seed)
  os.makedirs(args.out, exist_ok=True)
  tr_u8, tr_y, te_u8, te_y = load_cifar10_uint8(args.data, args.source)
  mean = torch.as_tensor(CIFAR_MEAN, device=dev, dtype=torch.float32)
  std = torch.as_tensor(CIFAR_STD, device=dev, dtype=torch.float32)
  tr_u8, te_u8 = torch.as_tensor(tr_u8, device=dev), torch.as_tensor(te_u8, device=dev)
  tr_y, te_y = torch.as_tensor(tr_y, device=dev), torch.as_tensor(te_y, device=dev)
  padded = F.pad(tr_u8.permute(0, 3, 1, 2).float(), (4, 4, 4, 4), mode='reflect').permute(0, 2, 3, 1).to(torch.uint8)
  test_x, train_eval_x = standardize(te_u8, mean, std), standardize(tr_u8[:args.train_eval], mean, std)

  K, B = len(args.lams), args.batch
  model = StackedResMLP(K, seed=args.seed, device=dev)
  params = model.params()
  lam = torch.tensor(args.lams, device=dev)
  n = tr_u8.shape[0]
  spe = n // B
  total = spe * args.epochs if args.max_steps is None else min(args.max_steps, spe * args.epochs)
  warm = spe * args.warmup_epochs
  gen = torch.Generator(device=dev).manual_seed(args.seed)

  x_buf = torch.zeros(B, 3072, device=dev)
  y_buf = torch.zeros(B, dtype=torch.long, device=dev)
  lr_buf = torch.zeros((), device=dev)
  stats = torch.zeros(2, K, device=dev)          # running sums since the last log: loss, correct

  def step_body():
    train_step(model, params, x_buf, y_buf, lam, lr_buf, stats)

  graph = None
  if not args.no_graph and dev.type == 'cuda':
    x_buf.copy_(standardize(tr_u8[:B], mean, std)), y_buf.copy_(tr_y[:B]), lr_buf.fill_(0.0)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                   # warmup with lr 0 (no parameter change)
      for _ in range(3):
        for q in params:
          q.grad = None
        step_body()
    torch.cuda.current_stream().wait_stream(s)
    for q in params:
      q.grad = None
    try:
      graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):
        step_body()
    except Exception as e:                       # e.g. an op without graph support: train eagerly (slower, same math)
      print(f'[delta] CUDA graph capture failed ({type(e).__name__}: {e}); eager steps', flush=True)
      graph = None
      for q in params:
        q.grad = None
    stats.zero_()

  log = open(os.path.join(args.out, 'log.jsonl'), 'w')
  meta = vars(args) | {'K': K, 'steps_per_epoch': spe, 'total_steps': total, 'device': str(dev),
                       'gpu': torch.cuda.get_device_name() if dev.type == 'cuda' else 'cpu', 'torch': torch.__version__,
                       'cuda_graph': graph is not None}
  json.dump(meta, open(os.path.join(args.out, 'config.json'), 'w'), indent=1)
  t0, step, since = time.time(), 0, 0
  for epoch in range(args.epochs):
    if step >= total:
      break
    epoch_x = standardize(gpu_augment(padded, gen), mean, std)
    perm = torch.randperm(n, generator=gen, device=dev)
    for i in range(spe):
      if step >= total:
        break
      idx = perm[i * B:(i + 1) * B]
      x_buf.copy_(epoch_x[idx]), y_buf.copy_(tr_y[idx])
      lr_buf.fill_(lr_at(step, total, warm, args.lr))
      if graph is not None:
        graph.replay()
      else:
        for q in params:
          q.grad = None
        step_body()
      step += 1
      since += 1
    done = epoch + 1
    if done % args.eval_every == 0 or step >= total or done == 1:
      run = (stats / torch.tensor([[since], [since * B]], device=dev)).tolist()
      stats.zero_()
      since = 0
      te_l, te_a = evaluate(model, test_x, te_y)
      tr_l, tr_a = evaluate(model, train_eval_x, tr_y[:args.train_eval])
      rec = {'epoch': done, 'step': step, 'lr': lr_at(step - 1, total, warm, args.lr), 'lams': args.lams,
             'run_train_loss': run[0], 'run_train_acc': run[1], 'test_loss': te_l, 'test_acc': te_a,
             'train_loss': tr_l, 'train_acc': tr_a, 'wall_sec': time.time() - t0}
      log.write(json.dumps(rec) + '\n')
      log.flush()
      el = time.time() - t0
      print(f'[delta] epoch {done} step {step}/{total} test ' + ' '.join(f'{a:.4f}' for a in te_a)
            + ' | train ' + ' '.join(f'{a:.3f}' for a in tr_a)
            + f' | {el:.0f}s, {1000 * el / step:.2f} ms/step, eta {el / step * (total - step) / 60:.0f} min', flush=True)
      if not all(math.isfinite(v) for v in te_l):
        print('[delta] non-finite test loss in some model', flush=True)
  log.close()
  recs = [json.loads(l) for l in open(os.path.join(args.out, 'log.jsonl'))]
  json.dump({'lams': args.lams, 'final_test_acc': recs[-1]['test_acc'], 'final_train_acc': recs[-1]['train_acc'],
             'best_test_acc': np.max([r['test_acc'] for r in recs], 0).tolist(), 'elapsed': time.time() - t0,
             'steps': step}, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
  print('[delta] done', flush=True)


if __name__ == '__main__':
  main()
