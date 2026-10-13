"""Optimizers, online train/eval steps and the experiment runner (notebook Section 4.1), with file logging."""

import dataclasses as dc
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rcdl.data import augment_epoch, standardize
from rcdl.layers import RCDLConfig
from rcdl.models import CifarRCDLClassifier


class MultiOptimizer:
  """Steps several torch optimizers as one."""

  def __init__(self, opts):
    self.opts = opts

  def zero_grad(self, set_to_none: bool = True):
    for o in self.opts:
      o.zero_grad(set_to_none=set_to_none)

  def step(self):
    for o in self.opts:
      o.step()


def create_rcdl_optimizer(model: CifarRCDLClassifier, cfg: RCDLConfig):
  """Optimizer over all nn.Parameters: the dense weights ('standard') or the stored tables ('hybrid', 'parametric_softmax').

  The growing layers ('gws', 'gauge', 'rbf') have no trainable parameters, so None is returned for them.
  """
  params = [p for p in model.parameters() if p.requires_grad]
  if not params:
    return None
  opt_name, lr = cfg.optimizer.lower(), cfg.learning_rate
  if opt_name == 'muon':            # Muon on the weight matrices, Adam on the biases
    mats, rest = [p for p in params if p.ndim == 2], [p for p in params if p.ndim != 2]
    return MultiOptimizer([torch.optim.Muon(mats, lr=lr, weight_decay=cfg.weight_decay)]
                          + ([torch.optim.Adam(rest, lr=lr)] if rest else []))
  if opt_name == 'adamw':
    return torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
  if opt_name == 'sgd':
    return torch.optim.SGD(params, lr=lr, momentum=cfg.model_momentum)
  if opt_name == 'rmsprop':         # decay 0.9, eps 1e-8
    return torch.optim.RMSprop(params, lr=lr, alpha=0.9, eps=1e-8)
  return torch.optim.Adam(params, lr=lr)


def rcdl_train_step(model: CifarRCDLClassifier, opt, images: torch.Tensor, labels: torch.Tensor,
                    input_grad: bool = False):
  """Single online training step for all RCDL models and baselines.

  input_grad=True reproduces the notebook literally (images.requires_grad_), which also backpropagates through the
  first layer's retrieval. The default skips that unused input gradient; the stored tuples are identical (tests).
  """
  model.train()
  logits = model(images.requires_grad_(True) if input_grad else images)
  loss = F.cross_entropy(logits, labels)
  if opt is not None:
    opt.zero_grad()
  loss.backward()                               # the RCDL layers' gradient hooks compute the new tuples here ...
  if opt is not None:
    opt.step()
  for layer in model.rcdl_layers().values():
    layer.commit()                              # ... which are appended to the databases at the end of the step
  return loss.item(), (logits.argmax(-1) == labels).float().mean().item()


@torch.no_grad()
def rcdl_evaluate(model: CifarRCDLClassifier, images: torch.Tensor, labels: torch.Tensor,
                  batch_size: int = 500) -> Tuple[float, float]:
  """Mean test loss and accuracy (eval mode: no database growth)."""
  model.eval()
  total_loss = total_correct = 0.0
  for i in range(0, len(images), batch_size):
    logits = model(images[i:i + batch_size])
    total_loss += F.cross_entropy(logits, labels[i:i + batch_size], reduction='sum').item()
    total_correct += (logits.argmax(-1) == labels[i:i + batch_size]).sum().item()
  return total_loss / len(images), total_correct / len(images)


def get_per_layer_db_sizes(model: CifarRCDLClassifier, cfg: RCDLConfig) -> Dict[str, int]:
  """Stored entries per RCDL layer (for 'standard': the fixed width hidden_dim of each dense block)."""
  if cfg.kernel_type == 'standard':
    return {f'layer_{i}': cfg.hidden_dim for i in range(cfg.num_layers)}
  return {name: layer.size for name, layer in model.rcdl_layers().items()}


def run_rcdl_experiment(cfg: RCDLConfig, data, out_dir: Optional[str] = None, eval_every: int = 250,
                        eval_subset: Optional[int] = None, eval_batch_size: int = 500, input_grad: bool = False,
                        train_u8: Optional[np.ndarray] = None, max_steps: Optional[int] = None,
                        log_every: int = 100, keep_model: bool = False) -> Dict[str, Any]:
  """Online CIFAR-10 runner for all model kinds, with the notebook's stream order (torch.randperm, seed = init seed).

  data: (train_x, train_y, test_x, test_y) tensors. eval_subset: intermediate evaluations on the first `eval_subset`
  test images (the final evaluation always uses the full test set). train_u8: raw images, required if cfg.augment.
  Writes config.json, evals.jsonl, train_curve.npz and summary.json to out_dir if given.
  """
  train_x, train_y, test_x, test_y = data
  torch.manual_seed(cfg.global_init_seed)
  model = CifarRCDLClassifier(cfg).to(train_x.device)
  opt = create_rcdl_optimizer(model, cfg)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
      json.dump(dc.asdict(cfg) | {'eval_every': eval_every, 'eval_subset': eval_subset, 'input_grad': input_grad,
                                  'max_steps': max_steps, 'torch': torch.__version__,
                                  'threads': torch.get_num_threads(), 'device': str(train_x.device)}, f, indent=1)
    eval_log = open(os.path.join(out_dir, 'evals.jsonl'), 'w')

  n, B = len(train_x), cfg.batch_size
  steps_per_epoch = n // B
  total_steps = steps_per_epoch * cfg.num_epochs if max_steps is None else min(max_steps, steps_per_epoch * cfg.num_epochs)
  layer_names = list(get_per_layer_db_sizes(model, cfg))
  total_full_cap = max(1, len(layer_names)) * cfg.ref_capacity
  train_losses, train_accs, evals = [], [], []
  start_time = time.time()

  def evaluate_and_log(step: int, final: bool = False):
    m = len(test_x) if (final or eval_subset is None) else eval_subset
    t0 = time.time()
    test_loss, test_acc = rcdl_evaluate(model, test_x[:m], test_y[:m], eval_batch_size)
    total = sum(get_per_layer_db_sizes(model, cfg).values())
    rec = {'step': step, 'samples': step * B, 'n_test': m, 'test_loss': test_loss, 'test_acc': test_acc,
           'capacity': total, 'capacity_frac': total / total_full_cap, 'eval_sec': time.time() - t0,
           'wall_sec': time.time() - start_time}
    evals.append(rec)
    if out_dir:
      eval_log.write(json.dumps(rec) + '\n')
      eval_log.flush()
    print(f'[eval] samples={step * B:,} n_test={m} loss={test_loss:.4f} acc={test_acc:.4f} '
          f'cap={total:,}/{total_full_cap:,} ({rec["eval_sec"]:.1f}s)', flush=True)

  evaluate_and_log(0, final=False)
  step, perm, order_gen = 0, None, torch.Generator().manual_seed(cfg.global_init_seed)
  aug_gen = np.random.default_rng(cfg.global_init_seed)
  for epoch in range(cfg.num_epochs):
    perm = torch.randperm(n, generator=order_gen).to(train_x.device)
    epoch_x = train_x
    if cfg.augment:
      epoch_x = torch.as_tensor(standardize(augment_epoch(train_u8, aug_gen)), device=train_x.device)
    for batch_idx in range(steps_per_epoch):
      if step >= total_steps:
        break
      idx = perm[batch_idx * B:(batch_idx + 1) * B]
      loss, acc = rcdl_train_step(model, opt, epoch_x[idx], train_y[idx], input_grad=input_grad)
      step += 1
      train_losses.append(loss)
      train_accs.append(acc)
      if step % log_every == 0:
        el = time.time() - start_time
        print(f'[train] step={step}/{total_steps} loss(avg{log_every})={np.mean(train_losses[-log_every:]):.4f} '
              f'acc={np.mean(train_accs[-log_every:]):.3f} elapsed={el:.0f}s eta={el / step * (total_steps - step):.0f}s',
              flush=True)
      if step % eval_every == 0 and step != total_steps:
        evaluate_and_log(step)
  evaluate_and_log(step, final=True)

  elapsed = time.time() - start_time
  final_dbs = get_per_layer_db_sizes(model, cfg)
  result = {'final_test_acc': evals[-1]['test_acc'], 'final_test_loss': evals[-1]['test_loss'],
            'final_db_size': sum(final_dbs.values()), 'total_full_capacity': total_full_cap,
            'per_layer_db': final_dbs, 'elapsed': elapsed, 'steps': step,
            'final_train_loss_avg100': float(np.mean(train_losses[-100:]))}
  print(f'--> Finished in {elapsed:.1f}s | Final Test Accuracy: {result["final_test_acc"] * 100:.2f}%', flush=True)
  if out_dir:
    eval_log.close()
    np.savez(os.path.join(out_dir, 'train_curve.npz'), loss=np.asarray(train_losses), acc=np.asarray(train_accs),
             perm=perm.cpu().numpy() if perm is not None else np.zeros(0))
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
      json.dump(result, f, indent=1)
  result['evals'] = evals
  if keep_model:
    result['model'] = model
  return result
