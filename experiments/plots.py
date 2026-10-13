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
    r['seed_dir'] = d
    ev = os.path.join(d, 'evals.jsonl')
    r['evals'] = [json.loads(l) for l in open(ev)] if os.path.exists(ev) else []
    cfg = os.path.join(d, 'config.json')
    r['config'] = json.load(open(cfg)) if os.path.exists(cfg) else {}
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


def fig_teacher_student(ts, out):
  best = {}
  for path, r in ts.items():
    st = r['cfg']['student']
    if st not in best or r['final_test_mse'] < best[st][1]['final_test_mse'] or 'full' in path:
      if st in best and 'full' in best[st][0] and 'full' not in path:
        continue
      best[st] = (path, r)
  if not best:
    return
  names = {'gws': 'Weighted softmax (growing)', 'mlp': 'Reference MLP', 'psm': 'Parametric softmax'}
  targets = {'gws': 2.6e-4, 'mlp': 1.7e-4, 'psm': 4e-3}
  fig, ax = new_fig(6.4, 4.0)
  for c, st in zip(SERIES, ('gws', 'mlp', 'psm')):
    if st not in best:
      continue
    path, r = best[st]
    xs = [p['step'] for p in r['log'] if p['step'] > 0]      # step 0 omitted: the initial rows' outputs are B x scale
    ys = [p['test_mse'] for p in r['log'] if p['step'] > 0]
    ax.plot(xs, ys, color=c, linewidth=2, label=f'{names[st]} ({os.path.basename(path)})')
    ax.scatter([200000], [targets[st]], marker='D', s=36, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
  ax.set_yscale('log')
  style(ax, 'Teacher-student regression: test MSE (diamonds: paper at 200k steps)', 'training step', 'test MSE')
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
  style(ax, 'Hybrid layer, one pass: test accuracy vs final capacity', 'final capacity (% of one row per sample)',
        'test accuracy (%)')
  ax.legend(fontsize=8, frameon=False, loc='lower right')
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'hybrid_capacity.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def fig_multi_epoch(runs, out):
  keys = [k for k in runs if k.endswith('_50ep') and not k.endswith('c100')]
  if not keys:
    return
  fig, ax = new_fig(7.0, 4.2)
  for c, k in zip(SERIES, sorted(keys)):
    r = runs[k][0]
    spe = 781
    ev = [e for e in r['evals'] if e['step'] > 0]
    ax.plot([e['step'] / spe for e in ev], [100 * e['test_acc'] for e in ev], '-o', markersize=3, linewidth=1.6,
            color=c, label=f'{k}: {100 * r["final_test_acc"]:.1f}')
  style(ax, 'CIFAR-10, 50 epochs (aug.): test accuracy (%; intermediate points on 2,000 test images)', 'epoch',
        'test accuracy (%)')
  ax.legend(fontsize=7, frameon=False, ncol=2)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'multi_epoch.png'), dpi=150, facecolor=SURFACE)
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
  if not facts:
    return
  Ns = sorted({r['cfg']['n_people'] for r in facts.values()})
  fig, axes = new_fig(4.2 * len(Ns), 3.8, ncols=len(Ns))
  axes = np.atleast_1d(axes)
  for ax, N in zip(axes, Ns):
    items = sorted((p, r) for p, r in facts.items() if r['cfg']['n_people'] == N)
    for c, (p, r) in zip(SERIES, items):
      ax.plot([q['step'] for q in r['log']], [q['loss'] for q in r['log']], linewidth=1.8, color=c,
              label=os.path.basename(p).replace(f'_N{N}', ''))
    K = items[0][1]['cfg']['n_values']
    ax.axhline(math.log(K), color=INK2, linewidth=0.8, linestyle=':')
    style(ax, f'N = {N} individuals: fact loss', 'training step', 'cross-entropy on all facts')
    ax.legend(fontsize=7, frameon=False)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'facts_plateau.png'), dpi=150, facecolor=SURFACE)
  plt.close(fig)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--results', nargs='*', default=['results/colab_a100', 'results/colab_l4b', 'results/local_facts'])
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
    facts.update(load_json_results(root, 'facts'))
  if os.path.exists(args.recovered):     # single-pass runs whose files were lost (see the file's comment)
    for k, accs in json.load(open(args.recovered)).items():
      if not k.startswith('_') and len(runs.get(k, [])) < len(accs):
        runs[k] = [{'final_test_acc': a} for a in accs]
  fig_singlepass(runs, baselines, args.out)
  fig_beta_eta(runs, args.out)
  fig_teacher_student(ts, args.out)
  fig_hybrid(runs, args.out)
  fig_multi_epoch(runs, args.out)
  fig_facts(facts, args.out)
  fig_train_loss(args.results + ['results/hf', 'results/cpu_binary'], args.out)
  print('figures:', sorted(os.listdir(args.out)))


if __name__ == '__main__':
  main()
