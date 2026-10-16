"""Figures for docs/REPRODUCTION.md from every result file under results/ (local runs and runs pulled from Colab).

Usage: python -m experiments.plots [--results results] [--out docs/figures]
Each figure is drawn only if its data exist. Paper targets are values read off the paper's plots (see the configs).
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Reference palette (dataviz skill, light mode): categorical slots in fixed order, sequential blue ramp, text inks.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
ORANGES = ['#f4b088', '#ee8a52', '#d4581f', '#9e3f12']    # sequential ramp of the second categorical hue
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def style(ax, title=None, xlabel=None, ylabel=None):
  ax.set_facecolor(SURFACE)
  for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
  for s in ('left', 'bottom'):
    ax.spines[s].set_color(INK2)
    ax.spines[s].set_linewidth(0.8)
  ax.tick_params(colors=INK2, labelsize=9)
  ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
  ax.set_axisbelow(True)
  if title:
    ax.set_title(title, color=INK, fontsize=11, loc='left')
  if xlabel:
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
  if ylabel:
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def new_fig(w=7.0, h=4.0, ncols=1):
  fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE)
  return fig, axes


def load_runs(root):
  """{experiment name: [summary dicts with 'evals' and 'curve' attached]} for every seed*/summary.json under root."""
  runs = defaultdict(list)
  for s in glob.glob(os.path.join(root, '**', 'seed*', 'summary.json'), recursive=True):
    d = os.path.dirname(s)
    r = json.load(open(s))
    if 'skipped' in r:                    # placeholder written to make --skip-done skip a run (GPU budget)
      continue
    r['seed_dir'] = d
    ev = os.path.join(d, 'evals.jsonl')
    r['evals'] = [json.loads(l) for l in open(ev)] if os.path.exists(ev) else []
    cfg = os.path.join(d, 'config.json')
    r['config'] = json.load(open(cfg)) if os.path.exists(cfg) else {}
    if r['config'].get('max_steps') is not None:   # truncated speed A/B runs (docs/REPRODUCTION.md section 10)
      continue
    runs[os.path.basename(os.path.dirname(d))].append(r)
  return runs


def load_json_results(root, sub):
  out = {}
  for f in glob.glob(os.path.join(root, '**', sub, '**', 'result.json'), recursive=True):
    out[os.path.relpath(os.path.dirname(f), root)] = json.load(open(f))
  return out


def acc_stats(rs):
  a = np.array([r['final_test_acc'] for r in rs]) * 100
  return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)


def fig_singlepass(runs, baselines, out):
  rows = [('GWS', 'gws', 59.4), ('RBF', 'rbf', 58.8), ('Gauge GWS', 'gauge', 58.4), ('MLP + Muon', 'mlp_muon', 57.5),
          ('MLP + AdamW', 'mlp_adamw', 50.0), ('MLP + Adam', 'mlp_adam', 49.5), ('MLP + RMSprop', 'mlp_rmsprop', 49.5),
          ('MLP + SGD', 'mlp_sgd', 46.0)]
  for pre, label, target in (('mesa_L4', 'MLP + Mesa (best)', 49.0), ('delta_L4', 'MLP + Delta (best)', 47.0)):
    cands = {k: v for k, v in runs.items() if k.startswith(pre)}
    if cands:
      best = max(cands, key=lambda k: acc_stats(cands[k])[0])
      rows.append((label, best, target))
  rows += [('Param. softmax 10%', 'psm10', None), ('Param. softmax 100%', 'psm100', None)]
  data = [(lab, *acc_stats(runs[k]), t) for lab, k, t in rows if k in runs]
  for key, lab, t in (('krr_rbf', 'Kernel ridge, RBF', 58.5), ('krr_exp', 'Kernel ridge, GWS kernel', None),
                      ('nw_exp', 'Retrieval vote (NW)', None), ('knn', 'kNN', 35.5)):
    if key in baselines:
      data.append((lab, baselines[key]['best_test'] * 100, 0.0, 1, t))
  if not data:
    return
  data.sort(key=lambda r: r[1])
  fig, ax = new_fig(7.2, 0.36 * len(data) + 1.2)
  y = np.arange(len(data))
  ax.barh(y, [d[1] for d in data], xerr=[d[2] for d in data], height=0.6, color=SERIES[0], zorder=2,
          error_kw={'ecolor': INK2, 'elinewidth': 1, 'capsize': 2}, label='This reproduction (mean ± std over seeds)')
  ty = [i for i, d in enumerate(data) if d[4] is not None]
  ax.scatter([data[i][4] for i in ty], ty, marker='D', s=36, color=SERIES[1], edgecolor=SURFACE, linewidth=1.5,
             zorder=3, label='Paper / notebook target')
  for i, d in enumerate(data):
    x = max(d[1] + d[2], d[4] or 0) + 0.5
    ax.text(x, i, f'{d[1]:.1f}' + (f' (n={d[3]})' if d[3] > 1 else ''), va='center', fontsize=8, color=INK)
  ax.set_yticks(y, [d[0] for d in data], fontsize=9, color=INK)
  ax.set_xlim(30, 64)
  style(ax, 'CIFAR-10, one pass over the training set: test accuracy (%)', 'test accuracy (%)')
  ax.legend(loc='lower right', fontsize=8, frameon=False)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'singlepass.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def fig_beta_eta(runs, out):
  etas, betas = [0.0, 0.003, 0.03, 0.3], [1.0, 3.0, 10.0, 30.0]
  grid = np.full((4, 4), np.nan)
  for i, e in enumerate(etas):
    for j, b in enumerate(betas):
      k = f'gws_L0_T10_5ep_eta{e:g}_beta{b:g}'
      if k in runs:
        grid[i, j] = acc_stats(runs[k])[0]
  if np.isnan(grid).all():
    return
  from matplotlib.colors import LinearSegmentedColormap
  cmap = LinearSegmentedColormap.from_list('blues', BLUES)
  fig, ax = new_fig(5.4, 4.0)
  im = ax.imshow(grid, cmap=cmap, vmin=35, vmax=57)
  for i in range(4):
    for j in range(4):
      if not np.isnan(grid[i, j]):
        r = betas[j] / etas[i] if etas[i] else math.inf
        ratio = '∞' if math.isinf(r) else (f'{r:,.0f}' if r >= 100 else f'{r:.3g}')
        ax.text(j, i, f'{grid[i, j]:.1f}\nβ/η={ratio}', ha='center', va='center', fontsize=8,
                color=SURFACE if grid[i, j] > 49 else INK)
  ax.set_xticks(range(4), [f'{b:g}' for b in betas])
  ax.set_yticks(range(4), [f'{e:g}' for e in etas])
  style(ax, 'GWS, L=0, T=10, 5 epochs: test accuracy (%)', 'value learning rate β', 'weight learning rate η')
  ax.grid(False)
  fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=8, colors=INK2)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'beta_eta_sweep.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def fig_teacher_student(ts, out, partial=None):
  """Final 200k-step runs where present (sweeps otherwise: best grid point per student at 10k steps)."""
  ts = {p: r for p, r in ts.items() if 'precision_check' not in p}
  is_long = lambda path: 'sweep' not in path.split(os.sep)[-2]       # experiment folder: *_sweep* = 10k-step grids
  curves = {}
  for path, r in ts.items():
    st, final = r['cfg']['student'], is_long(path)
    key = path.split(os.sep)[-2] if final else st
    if final or not any(st == r2['cfg']['student'] and is_long(p2) for p2, r2 in ts.items()):
      if key not in curves or r['final_test_mse'] < curves[key][1]['final_test_mse']:
        curves[key] = (path, r)
  if not curves:
    return
  names = {'gws': 'Weighted softmax (growing)', 'mlp': 'Reference MLP', 'psm': 'Parametric softmax'}
  targets = {'gws': 2.6e-4, 'mlp': 1.7e-4, 'psm': 4e-3}
  color = dict(zip(('gws', 'mlp', 'psm'), SERIES))            # one color per student: curves and paper markers
  fig, ax = new_fig(6.6, 4.0)
  for key in sorted(curves):
    path, r = curves[key]
    st = r['cfg']['student']
    xs = [p['step'] for p in r['log'] if p['step'] > 0]      # step 0 omitted: the initial rows' outputs are B x scale
    ys = [p['test_mse'] for p in r['log'] if p['step'] > 0]
    dashed = st == 'mlp' and r['cfg']['batch_size'] == 64
    label = f"{names[st]}, batch {r['cfg']['batch_size']}, {xs[-1] // 1000}k steps: {ys[-1]:.2e}"
    ax.plot(xs, ys, color=color[st], linewidth=2, linestyle='--' if dashed else '-', label=label)
  if partial and not any(r['cfg']['student'] == 'gws' and r['log'][-1]['step'] >= 200000 for _, r in curves.values()):
    # GWS 200k run stopped at 40k (log only); drawn only while no complete 200k GWS run exists
    ax.plot([q[0] for q in partial], [q[1] for q in partial], color=color['gws'], linewidth=2, linestyle=':',
            label=f'{names["gws"]}, 200k run stopped at {partial[-1][0] // 1000}k: {partial[-1][1]:.2e}')
  for st in ('gws', 'mlp', 'psm'):
    ax.scatter([200000], [targets[st]], marker='D', s=36, color=color[st], edgecolor=SURFACE, linewidth=1.5,
               zorder=3, label=f'paper at 200k steps, {names[st]} (read off plot)')
  ax.set_xscale('log')
  ax.set_yscale('log')
  style(ax, 'Teacher-student regression: test MSE', 'training step', 'test MSE')
  ax.legend(fontsize=7, frameon=False)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'teacher_student.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def fig_hybrid(runs, out):
  paper = {35: (88, 58.6), 40: (79, 58.5), 45: (65, 58.0), 50: (45, 56.5), 55: (26, 54.5), 60: (5, 50.0),
           65: (1, 41.0), 70: (0, 34.5)}
  pts = []
  for tau in paper:
    k = f'hybrid_tau{tau}'
    if k in runs:
      r = runs[k][0]
      pts.append((tau, 100 * r['final_db_size'] / r['total_full_capacity'], 100 * r['final_test_acc']))
  if not pts:
    return
  fig, ax = new_fig(6.0, 4.0)
  ax.plot([p[1] for p in pts], [p[2] for p in pts], '-o', color=SERIES[0], markersize=6, linewidth=2,
          label='This reproduction', zorder=3)
  ax.plot([v[0] for v in paper.values()], [v[1] for v in paper.values()], '--D', color=SERIES[1], markersize=5,
          linewidth=1.2, label='Paper Fig. 6 (read off plot)', zorder=2)
  for tau, x, yv in pts:
    ax.annotate(f'τ={tau}', (x, yv), textcoords='offset points', xytext=(4, -10), fontsize=7, color=INK2)
  tuned = [rs[0] for k, rs in runs.items() if k.startswith('hybrid_tau35_T')]   # hybrid_tau35_tuning.json
  if tuned:
    b = max(tuned, key=lambda r: r['final_test_acc'])
    ax.scatter([100 * b['final_db_size'] / b['total_full_capacity']], [100 * b['final_test_acc']], marker='*', s=140,
               color=SERIES[2], edgecolor=SURFACE, linewidth=1.0, zorder=4,
               label=f'τ=35, best of {len(tuned) + 1} settings (paper grid): {100 * b["final_test_acc"]:.1f}')
  style(ax, 'Hybrid layer, one pass: test accuracy vs final capacity', 'final capacity (% of one row per sample)',
        'test accuracy (%)')
  ax.legend(fontsize=8, frameon=False, loc='lower right')
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'hybrid_capacity.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def fig_multi_epoch(runs, out):
  full = lambda k: 'sub5k' not in k                         # 5,000-image runs: fig_fig5_scaled
  panels = [('CIFAR-10', [k for k in runs if k.endswith(('_50ep', '_50ep_bf16')) and full(k)]),
            ('CIFAR-100', [k for k in runs if k.endswith(('_50ep_c100', '_50ep_c100_bf16')) and full(k)])]
  panels = [(t, ks) for t, ks in panels if ks]
  if not panels:
    return
  fig, axes = new_fig(6.4 * len(panels), 4.2, ncols=len(panels))
  for ax, (title, keys) in zip(np.atleast_1d(axes), panels):
    best = {k: max(runs[k], key=lambda r: r['final_test_acc']) for k in keys}   # gws_L0_T10_50ep: eta 1 and 0.003 runs
    method = lambda k: '_'.join(k.split('_')[:2])          # mlp_mesa, mlp_muon, gws_L2, ...: one curve per method
    top = {}
    for k in keys:
      if method(k) not in top or best[k]['final_test_acc'] > best[top[method(k)]]['final_test_acc']:
        top[method(k)] = k
    keys = list(top.values())                              # tuning runs: docs/REPRODUCTION.md tables
    for c, k in zip(SERIES * 2, sorted(keys, key=lambda k: -best[k]['final_test_acc'])):
      r = best[k]
      spe = 781
      ev = [e for e in r['evals'] if e['step'] > 0]
      ax.plot([e['step'] / spe for e in ev], [100 * e['test_acc'] for e in ev], '-o', markersize=3, linewidth=1.6,
              color=c, label=f'{k}: {100 * r["final_test_acc"]:.1f}')
    style(ax, f'{title}, 50 epochs (aug.): test accuracy (%)', 'epoch', 'test accuracy (%)')
    ax.legend(fontsize=7, frameon=False)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'multi_epoch.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def fig_overfit(runs, out):
  """Reduced-scale paper Fig. 5 (CPU runs): train (dashed) and test (solid) accuracy against epoch, 5,000 images."""
  keys = sorted(k for k in runs if 'sub5k' in k and k.endswith(('_40ep', '_100ep')))
  if not keys:
    return
  fig, ax = new_fig(7.0, 4.2)
  for c, k in zip(SERIES, keys):
    r = runs[k][0]
    spe = r['config'].get('train_subset', 50000) // r['config'].get('batch_size', 64)
    ev = [e for e in r['evals'] if e['step'] > 0]
    ep = [e['step'] / spe for e in ev]
    ax.plot(ep, [100 * e['test_acc'] for e in ev], '-o', markersize=3, linewidth=1.8, color=c,
            label=f'{k}: test {100 * r["final_test_acc"]:.1f}')
    ax.plot(ep, [100 * e['train_acc'] for e in ev], '--', linewidth=1.4, color=c, label=f'{k}: train')
  style(ax, 'CIFAR-10, 5,000 training images: train (dashed) vs test (solid) accuracy', 'epoch', 'accuracy (%)')
  ax.legend(fontsize=7, frameon=False)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'overfit_subset.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def fig_fig5_scaled(runs, out):
  """Paper Fig. 5 with the paper's ratios at 1/10 of the data: x = training progress (%), parametric softmax 2,000
  epochs, GWS 50 epochs; train (dashed) and test (solid) accuracy."""
  keys = sorted(k for k in runs if k.endswith(('sub5k_50ep', 'sub5k_2000ep')))
  if not keys:
    return
  fig, ax = new_fig(7.0, 4.2)
  for c, k in zip(SERIES, keys):
    r = max(runs[k], key=lambda r: len(r['evals']))
    ev = [e for e in r['evals'] if e['step'] > 0 and 'train_acc' in e]
    total = max(e['step'] for e in ev)
    x = [100 * e['step'] / total for e in ev]
    ep = r['config'].get('num_epochs')
    name = ('Weighted softmax (GWS)' if k.startswith('gws') else
            f"Parametric softmax, Adam {r['config'].get('learning_rate'):g}") + f', {ep} epochs'
    ax.plot(x, [100 * e['test_acc'] for e in ev], '-', linewidth=1.8, color=c,
            label=f'{name}: test (final {100 * r["final_test_acc"]:.1f})')
    ax.plot(x, [100 * e['train_acc'] for e in ev], '--', linewidth=1.4, color=c, label=f'{name}: train')
  style(ax, 'Fig. 5 at 1/10 scale (5,000 images): train (dashed) and test (solid)', 'training progress (%)',
        'accuracy (%)')
  ax.legend(fontsize=7, frameon=False)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'fig5_scaled.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def fig_deltanet(roots, out):
  """Paper App. E.3.1 figure: train and test accuracy against epoch for each Delta-rule decay rate λ."""
  logs = [f for root in roots for f in glob.glob(os.path.join(root, '**', 'deltanet', 'log.jsonl'), recursive=True)]
  if not logs:
    return
  recs = [json.loads(l) for l in open(max(logs, key=os.path.getsize))]
  lams, ep = recs[0]['lams'], [r['epoch'] for r in recs]
  fig, axes = new_fig(12.0, 4.2, ncols=2)
  for i, lam in enumerate(lams):
    lab = f'λ = {lam:g}'
    axes[0].plot(ep, [100 * r['run_train_acc'][i] for r in recs], color=SERIES[i], linewidth=1.6, label=lab)
    axes[1].plot(ep, [100 * r['test_acc'][i] for r in recs], color=SERIES[i], linewidth=1.6,
                 label=f'{lab}: final {100 * recs[-1]["test_acc"][i]:.1f}, best {100 * max(r["test_acc"][i] for r in recs):.1f}')
  style(axes[0], 'Train accuracy (augmented batches, dropout on)', 'epoch', 'accuracy (%)')
  style(axes[1], 'Test accuracy', 'epoch', 'accuracy (%)')
  for ax in axes:
    ax.legend(fontsize=7, frameon=False)
  fig.suptitle('10-block residual MLP, SGD with Delta-rule decay (paper: λ = 0 → ~67%, λ = 1e-5 → ~74%)',
               color=INK, fontsize=10, x=0.01, ha='left')
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'deltanet_long.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def fig_train_loss(roots, out, names=('gws', 'rbf', 'gauge', 'mlp_muon', 'mlp_adamw', 'mlp_adam', 'mlp_sgd'),
                   prefixes=('mesa_L4', 'delta_L4'), window=100):
  """Online (prequential) training loss of single-pass runs: each batch is scored before the model learns from it."""
  curves = {}
  for root in roots:
    for f in glob.glob(os.path.join(root, '**', 'seed*', 'train_curve.npz'), recursive=True):
      name = os.path.basename(os.path.dirname(os.path.dirname(f)))
      cfg = os.path.join(os.path.dirname(f), 'config.json')
      if not os.path.exists(cfg) or json.load(open(cfg)).get('num_epochs', 1) != 1:
        continue
      if name in names or name.startswith(prefixes):
        curves.setdefault(name, f)
  for pre in prefixes:                     # keep only the best grid point per prefix (lowest final loss)
    cands = [k for k in curves if k.startswith(pre)]
    if len(cands) > 1:
      best = min(cands, key=lambda k: np.load(curves[k])['loss'][-window:].mean())
      for k in cands:
        if k != best:
          curves.pop(k)
  if not curves:
    return
  fig, ax = new_fig(7.0, 4.0)
  order = sorted(curves, key=lambda k: np.load(curves[k])['loss'][-window:].mean())
  for c, k in zip(SERIES, order):
    loss = np.load(curves[k])['loss']
    sm = np.convolve(loss, np.ones(window) / window, mode='valid')
    ax.plot(np.arange(window, len(loss) + 1) * 16, sm, color=c, linewidth=1.8, label=f'{k}: {sm[-1]:.3f}')
  ax.set_ylim(1.0, 2.6)
  style(ax, 'CIFAR-10, one pass: online training loss (moving average of 100 batches)', 'training samples seen',
        'cross-entropy')
  ax.legend(fontsize=7, frameon=False)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'train_loss_singlepass.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def fig_facts(facts, out):
  """Fact loss vs step per population size N (main phase), and old/new fact accuracy during fine-tuning."""
  main = {p: r for p, r in facts.items() if not r['cfg'].get('finetune_people')}
  if main:
    third = lambda p: 'molab' in p                          # third-round width / value-rate scans (molab)
    Ns = sorted({r['cfg']['n_people'] for r in main.values()})
    panels = [(f'N = {N} individuals: fact loss', [(p, r) for p, r in sorted(main.items())
                                                    if r['cfg']['n_people'] == N and not third(p)], None) for N in Ns]
    scan = {p: r for p, r in main.items() if r['cfg']['n_people'] == 16384 and
            (third(p) or os.path.basename(p) in ('cos_w16384_N16384', 'rbf_last_eta0.3_N16384', 'rbf_last_eta1_N16384',
                                                 'geglu_w512_N16384'))}
    if any(third(p) for p in scan):
      cos = sorted((p for p in scan if scan[p]['cfg']['ffn'] == 'cos'), key=lambda p: scan[p]['cfg']['width'])
      rbf = sorted((p for p in scan if scan[p]['cfg']['ffn'] == 'rbf'), key=lambda p: scan[p]['cfg']['rbf_eta'])
      geg = [p for p in scan if os.path.basename(p) == 'geglu_w512_N16384']
      ramp = {p: BLUES[2 + i] for i, p in enumerate(cos)} | {p: ORANGES[i] for i, p in enumerate(rbf)}
      ramp |= {p: INK2 for p in geg}
      panels.append(('N = 16384: Cos width (blue), RBF η (orange)',
                     [(p, scan[p]) for p in geg + cos + rbf], ramp))
    fig, axes = new_fig(4.8 * len(panels), 3.9, ncols=len(panels))
    for ax, (title, items, colors) in zip(np.atleast_1d(axes), panels):
      N = items[0][1]['cfg']['n_people']
      for c, (p, r) in zip(SERIES * 2, items):
        ax.plot([q['step'] for q in r['log']], [q['loss'] for q in r['log']], linewidth=1.8,
                color=colors[p] if colors else c,
                label=f"{os.path.basename(p).replace(f'_N{N}', '')}: acc {r['final_acc']:.3f}")
      K = items[0][1]['cfg']['n_values']
      ax.axhline(math.log(K), color=INK2, linewidth=0.8, linestyle=':')
      ax.axhline(0.9 * math.log(K), color=INK2, linewidth=0.6, linestyle='--')
      style(ax, title, 'training step', 'cross-entropy (dotted: ln K; dashed: 0.9 ln K)')
      ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'facts_plateau.png'), dpi=150, facecolor=SURFACE)
    plt.close(fig)
  ft = {p: r for p, r in facts.items() if r['cfg'].get('finetune_people')}
  if ft:
    fig, ax = new_fig(6.8, 4.0)
    for c, (p, r) in zip(SERIES, sorted(ft.items())):
      T0 = r['cfg']['steps']
      log = [q for q in r['log'] if q['step'] >= T0]
      name = os.path.basename(p)
      ax.plot([q['step'] - T0 for q in log], [q['acc'] for q in log], color=c, linewidth=2, label=f'{name}: old facts')
      ax.plot([q['step'] - T0 for q in log], [q['acc_new'] for q in log], color=c, linewidth=1.4, linestyle='--',
              label=f'{name}: new facts')
    style(ax, 'Fine-tuning on new individuals only (no replay): fact accuracy', 'fine-tuning step', 'accuracy')
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'facts_finetune.png'), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def load_bios(roots):
  """{experiment name: result.json} of every biography run (rcdl/bios.py) under <root>/**/bios/<name>/result.json, and
  the archived copies in docs/data/bios/<name>.json (results/ is not in git)."""
  out = {}
  for root in roots:
    for f in glob.glob(os.path.join(root, '**', 'bios', '*', 'result.json'), recursive=True):
      out[os.path.basename(os.path.dirname(f))] = json.load(open(f))
    for f in glob.glob(os.path.join(root, '**', 'bios', '*.json'), recursive=True):
      out.setdefault(os.path.basename(f)[:-len('.json')], json.load(open(f)))
  return out


BIOS_LABELS = {'geglu_N': 'GeGLU (all layers)', 'geglu_replay': 'GeGLU (all layers)', 'gegluf_last': 'routed GeGLU (last)', 'rbf_T2': 'growing RBF (last)',
               'cos_last_D64k': 'Cos-Bochner 64k (last)', 'cos_all_D16k': 'Cos-Bochner 16k (all)',
               'cos_all_D4k': 'Cos-Bochner 4k (all)', 'cos_all_D1k': 'Cos-Bochner 1k (all)'}


def _bios_label(name):
  label = next((v for k, v in BIOS_LABELS.items() if k in name), name)
  return label + (f", seed {name.split('seed')[1]}" if 'seed' in name else '')


MUON_RUNS = ('F_', 'G_', 'H_')    # section 7.5c: Muon, kernel width and mechanism runs (own figure)


def fig_bios(bios, out):
  """Attribute loss and accuracy against steps at N = 8,192 for every FFN variant; plateau against N."""
  bios = {k: r for k, r in bios.items() if not k.startswith(MUON_RUNS)}   # Muon and mechanism runs: fig_bios_muon
  n8 = {k: r for k, r in bios.items() if r['cfg']['n_people'] == 8192 and 'replay' not in k and 'seed' not in k
        and not k.startswith('E_')}          # E_*: diagnostic reruns (seen-template evaluation), shown in the text
  if not n8:
    return
  order = sorted(n8, key=lambda k: list(BIOS_LABELS).index(next((b for b in BIOS_LABELS if b in k), 'rbf_T2')))
  # one color per variant in a fixed order, so the same variant has the same color in every panel and figure
  fig, axes = new_fig(12.0, 4.2, ncols=3)
  for c, k in zip(SERIES, order):
    r = n8[k]
    pre = [e for e in r['log'] if e['step'] <= r['cfg']['steps']]
    x = [e['step'] for e in pre]
    axes[0].plot(x, [e['attr_loss'] for e in pre], color=c, linewidth=1.6, label=_bios_label(k))
    axes[1].plot(x, [100 * e['attr_acc'] for e in pre], color=c, linewidth=1.6, label=_bios_label(k))
  base = next(iter(n8.values()))['no_knowledge_loss']
  axes[0].axhline(base, color=INK2, linestyle='--', linewidth=1.0)
  axes[0].text(axes[0].get_xlim()[1], base, ' no knowledge', color=INK2, fontsize=8, va='bottom', ha='right')
  axes[0].set_yscale('log')
  style(axes[0], 'Attribute loss, N = 8,192 (held-out frames)', 'step', 'nats per attribute')
  style(axes[1], 'Attribute accuracy, N = 8,192', 'step', 'accuracy (%)')
  axes[1].legend(fontsize=7, frameon=False, loc='lower right')
  geglu = sorted(((r['cfg']['n_people'], r) for k, r in bios.items() if r['ffn_kinds'] == ['geglu'] * len(r['ffn_kinds'])
                  and 'replay' not in k and 'seed' not in k), key=lambda t: t[0])
  rbf = sorted(((r['cfg']['n_people'], r) for k, r in bios.items() if 'rbf' in r['ffn_kinds'] and r['cfg']['rbf_T'] == 2.0
                and 'seed' not in k), key=lambda t: t[0])     # N = 4,096: 3,000-step schedule (scan run)
  for c, (rows, lab) in zip(SERIES, ((geglu, 'GeGLU'), (rbf, 'growing RBF (last)'))):
    pts = [(n, r['acc50_step']) for n, r in rows if r['acc50_step'] is not None]
    if pts:
      axes[2].plot([p[0] for p in pts], [p[1] for p in pts], 'o-', color=c, markersize=7, linewidth=1.6, label=lab)
  axes[2].set_xscale('log', base=2)
  style(axes[2], 'Steps to 50% attribute accuracy against N', 'individuals N', 'step')
  axes[2].legend(fontsize=8, frameon=False)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'bios_plateau.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)
  ft = {k: r for k, r in bios.items() if r['cfg'].get('finetune_people')}
  if ft:
    fig, axes = new_fig(10.0, 4.0, ncols=2)
    for c, k in zip(SERIES, sorted(ft)):
      r = ft[k]
      post = [e for e in r['log'] if e['step'] >= r['cfg']['steps']]
      x = [e['step'] - r['cfg']['steps'] for e in post]
      lab = _bios_label(k) + (' + 50% replay' if 'replay' in k else '')
      axes[0].plot(x, [100 * e['attr_acc'] for e in post], color=c, linewidth=1.6, label=lab)
      axes[1].plot(x, [100 * e.get('new_attr_acc', float('nan')) for e in post], color=c, linewidth=1.6, label=lab)
    style(axes[0], 'Fine-tuning on 1,024 new individuals: old individuals', 'fine-tuning step', 'attribute accuracy (%)')
    style(axes[1], 'New individuals', 'fine-tuning step', 'attribute accuracy (%)')
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'bios_finetune.png'), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_fig5_full(roots, out):
  """Paper Fig. 5 at full scale: parametric softmax with 2.56M trained keys (bf16 table, fp32 Adam second moment,
  stochastic rounding), test and train accuracy against epochs. Reads evals.jsonl directly: the run is stopped long
  before its 2,000 epochs, so it never writes summary.json (load_runs needs that file)."""
  files = sorted(f for root in roots for f in glob.glob(os.path.join(root, '**', '*fullscale_sr', 'seed*', 'evals.jsonl'),
                                                         recursive=True))
  files += sorted(f for root in roots for f in glob.glob(os.path.join(root, '**', 'fig5_fullscale_evals.jsonl'),
                                                          recursive=True))   # archived copy in docs/data
  if not files:
    return
  ev = max(([json.loads(l) for l in open(f) if l.strip()] for f in files), key=len)   # the longest copy of the log
  ev = [e for e in ev if e['step'] > 0 and 'train_acc' in e]
  if not ev:
    return
  spe = 50000 // 64
  x = [e['step'] / spe for e in ev]
  fig, axes = new_fig(10.0, 5.0, ncols=2)
  axes[0].plot(x, [100 * e['test_acc'] for e in ev], '-', color=SERIES[1], linewidth=1.8,
               label='parametric softmax, 2.56M keys: test (10,000 images)')
  axes[0].plot(x, [100 * e['train_acc'] for e in ev], '--', color=SERIES[1], linewidth=1.4,
               label='parametric: train (2,000 non-augmented images)')
  gws = [[json.loads(l) for l in open(f) if l.strip()] for root in roots
         for f in glob.glob(os.path.join(root, '**', 'gws_L0_T10_50ep', 'seed*', 'evals.jsonl'), recursive=True)]
  gws = [g for g in gws if g and 'train_acc' in g[-1]]
  if gws:   # two runs share this name (eta 1 and 0.003); section 7.2 reports the better one
    g = [e for e in max(gws, key=lambda g: g[-1]['test_acc']) if e['step'] > 0]
    xg = [e['step'] / spe for e in g]
    axes[0].plot(xg, [100 * e['test_acc'] for e in g], '-', color=SERIES[0], linewidth=1.8, label='GWS, L = 0, 50 epochs: test (2,000 images; epoch 50: 10,000)')
    axes[0].plot(xg, [100 * e['train_acc'] for e in g], '--', color=SERIES[0], linewidth=1.4, label='GWS: train')
    axes[1].plot(xg, [e['test_loss'] for e in g], '-', color=SERIES[0], linewidth=1.8,
                 label='GWS: test (2,000 images; epoch 50: 10,000)')
  axes[0].plot([60], [55.5], 'o', color=INK2, markersize=7, label='paper: test peak (about 55.5%, about epoch 60)')
  axes[0].axhline(49.0, color=INK2, linestyle=':', linewidth=1.0, label='paper: test at epoch 2,000 (about 49%)')
  style(axes[0], 'Fig. 5 at full scale (all 50,000 images)', 'epoch', 'accuracy (%)')
  axes[1].plot(x, [e['test_loss'] for e in ev], '-', color=SERIES[1], linewidth=1.8, label='parametric: test (10,000 images)')
  axes[1].plot(x, [e['train_loss'] for e in ev], '--', color=SERIES[1], linewidth=1.4, label='parametric: train (non-augmented)')
  style(axes[1], 'Cross-entropy loss', 'epoch', 'nats')
  axes[1].legend(fontsize=8, frameon=False)
  fig.tight_layout(rect=(0, 0.16, 1, 1))
  fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8, frameon=False)
  fig.savefig(os.path.join(out, 'fig5_fullscale.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def fig_bios_muon(bios, out):
  """Section 7.5c: pre-training and fine-tuning with AdamW against Muon (colour = FFN, line style = optimizer), and the
  fine-tuning branches of runs H (same pre-trained state; table writes only when every weight is frozen)."""
  need = ['C_geglu_N8k', 'C_rbf_T2_N8k', 'G_geglu_muon_ft', 'G_cos_all_D4k_muon_ft', 'G_rbf_T2_muon_ft',
          'H_rbf_T2_adamw_mech', 'H_rbf_T2_muon_mech']
  if any(k not in bios for k in need):
    return
  col = {'geglu': SERIES[0], 'rbf': SERIES[1], 'cos': SERIES[2]}
  runs = [('C_geglu_N8k', 'geglu', '--', 'GeGLU, AdamW'), ('C_rbf_T2_N8k', 'rbf', '--', 'growing RBF (last), AdamW'),
          ('F_cos_all_D4k_T2', 'cos', '--', 'Cos-Bochner 4k (T = 2), AdamW'),
          ('G_geglu_muon_ft', 'geglu', '-', 'GeGLU, Muon'), ('G_rbf_T2_muon_ft', 'rbf', '-', 'growing RBF (last), Muon'),
          ('G_cos_all_D4k_muon_ft', 'cos', '-', 'Cos-Bochner 4k, Muon')]
  fig, axes = new_fig(15.0, 4.6, ncols=3)
  for k, kind, ls, lab in runs:
    if k not in bios:
      continue
    r = bios[k]
    T = r['cfg']['steps']
    pre = [e for e in r['log'] if e['step'] <= 3000]
    axes[0].plot([e['step'] for e in pre], [100 * e['attr_acc'] for e in pre], ls, color=col[kind], linewidth=1.6, label=lab)
    post = [e for e in r['log'] if e['step'] >= T]
    if r['cfg'].get('finetune_people') and post:
      axes[1].plot([e['step'] - T for e in post], [100 * e['attr_acc'] for e in post], ls, color=col[kind], linewidth=1.6,
                   label=lab)
  style(axes[0], 'Pre-training, N = 8,192 (held-out frames)', 'step', 'attribute accuracy (%)')
  axes[0].legend(fontsize=7, frameon=False, loc='lower right')
  style(axes[1], 'Fine-tuning on 1,024 new individuals: old facts', 'fine-tuning step', 'attribute accuracy (%)')
  branches = [('H_rbf_T2_adamw_mech', 'adamw', SERIES[1], '--', 'AdamW pre-training, AdamW fine-tuning'),
              ('H_rbf_T2_adamw_mech', 'frozen', SERIES[4], '--', 'AdamW pre-training, weights frozen'),
              ('H_rbf_T2_muon_mech', 'muon', SERIES[1], '-', 'Muon pre-training, Muon fine-tuning'),
              ('H_rbf_T2_muon_mech', 'adamw', SERIES[3], '-', 'Muon pre-training, AdamW fine-tuning'),
              ('H_rbf_T2_muon_mech', 'frozen', SERIES[4], '-', 'Muon pre-training, weights frozen')]
  for k, b, c, ls, lab in branches:
    v = bios[k].get('finetune_variants', {}).get(b)
    if not v:
      continue
    T = bios[k]['cfg']['steps']
    axes[2].plot([e['step'] - T for e in v['log']], [100 * e['attr_acc'] for e in v['log']], ls, color=c, linewidth=1.6,
                 label=lab)
  for k, ls in (('H_rbf_T2_adamw_mech', '--'), ('H_rbf_T2_muon_mech', '-')):   # table read off, end of pre-training
    v = next(iter(bios[k].get('finetune_variants', {}).values()), None)
    if v and 'no_table_attr_acc' in v['log'][0]:
      axes[2].plot([0], [100 * v['log'][0]['no_table_attr_acc']], 'o', color=INK2, markersize=6, fillstyle='none' if ls == '--' else 'full')
  axes[2].plot([], [], 'o', color=INK2, markersize=6, label='table read off at step 0 (filled: Muon, open: AdamW)')
  style(axes[2], 'Growing RBF: fine-tuning branches from one state (runs H)', 'fine-tuning step', 'old-fact accuracy (%)')
  axes[2].set_ylim(10, 101)
  axes[2].legend(fontsize=7, frameon=False, loc='lower right')
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'bios_muon.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--results', nargs='*', default=['results/colab_a100', 'results/colab_l4b', 'results/cpu_overfit',
                                                  'results/colab_a2', 'results/colab_g4', 'results/molab_pull',
                                                  'results/molab_pull2', 'docs/data'])
  p.add_argument('--recovered', default='docs/data/singlepass_recovered.json')
  p.add_argument('--out', default='docs/figures')
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)
  runs, baselines, ts, facts = defaultdict(list), {}, {}, {}
  for root in args.results:
    for k, v in load_runs(root).items():
      runs[k] += v
    for f in glob.glob(os.path.join(root, '**', 'baselines_n50000.json'), recursive=True):
      baselines.update({k: v for k, v in json.load(open(f)).items() if isinstance(v, dict)})
    ts.update(load_json_results(root, 'ts'))
    ts.update(load_json_results(root, 'ts_final'))
    facts.update({p: r for p, r in load_json_results(root, 'facts2').items()})   # task with 2-token names (v2)
  if os.path.exists(args.recovered):     # single-pass runs whose files were lost (see the file's comment)
    for k, accs in json.load(open(args.recovered)).items():
      if not k.startswith('_') and len(runs.get(k, [])) < len(accs):
        runs[k] = [{'final_test_acc': a} for a in accs]
  fig_singlepass(runs, baselines, args.out)
  fig_beta_eta(runs, args.out)
  partial = []                           # (step, test MSE) of the stopped 200k GWS run, from its log
  for root in args.results:
    for f in glob.glob(os.path.join(root, '**', 'ts_final_gws_partial.log'), recursive=True):
      lines = open(f).read().split('=== gws_final')[-1].splitlines()
      partial = [(int(l.split('step=')[1].split()[0]), float(l.split('test=')[1].split()[0]))
                 for l in lines if l.startswith('[ts:gws] step=')]
  fig_teacher_student(ts, args.out, partial)
  fig_hybrid(runs, args.out)
  fig_multi_epoch(runs, args.out)
  fig_overfit(runs, args.out)
  fig_fig5_scaled(runs, args.out)
  fig_fig5_full(args.results, args.out)
  bios = load_bios(args.results)
  fig_bios(bios, args.out)
  fig_bios_muon(bios, args.out)
  fig_deltanet(args.results, args.out)
  fig_facts(facts, args.out)
  fig_train_loss(args.results + ['results/hf', 'results/cpu_binary'], args.out)
  print('figures:', sorted(os.listdir(args.out)))


if __name__ == '__main__':
  main()
