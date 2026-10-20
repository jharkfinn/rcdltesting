"""Tables and verdicts for tests A, B and C of the looping follow-ups (docs/data/loops/predictions_abc.md).

Usage: python -m experiments.loops_abc_analysis --level DIR --gate DIR --why DIR [--level_long DIR] [--width_old DIR]
       [--infiloop_old DIR] [--out DIR]
Each DIR holds <name>/result.json; names end in _K{K}_s{seed}. Missing runs are reported as '-' and the rules that need
them as 'pending'. With --out, writes tables.md and summary.json (every number the tables and verdicts use).

Measures (as defined in predictions_abc.md):
  late accuracy: mean over the evaluations at steps >= 0.75 * steps;
  gain: late accuracy of a variant minus that of its baseline, same seed, in points;
  speed-up at threshold t: (step at which the baseline first reaches t) / (step at which the variant first reaches t),
    steps linearly interpolated between evaluations; early thresholds 0.2-0.5, high thresholds 0.7 and 0.9;
  the 30% point of a run: its first evaluation with accuracy >= 0.3 (the per-pass probe is read there).
"""

import argparse
import glob
import json
import os
import re
from statistics import mean

EARLY, HIGH = (0.2, 0.3, 0.4, 0.5), (0.7, 0.9)
SEEDS = (1, 2)


def load(root):
  runs = {}
  for f in sorted(glob.glob(os.path.join(root or '', '*', 'result.json'))) if root else []:
    runs[os.path.basename(os.path.dirname(f))] = json.load(open(f))
  return runs


def late(r, key='acc', frac=0.75):
  v = [e[key] for e in r['log'] if frac * r['cfg']['steps'] <= e['step'] <= r['cfg']['steps'] and key in e]
  return mean(v) if v else None


def tstep(r, t):
  """First step at which accuracy reaches t, linearly interpolated between evaluations (None: never reached)."""
  prev = None
  for e in r['log']:
    if e['step'] > r['cfg']['steps']:
      break
    if e['acc'] >= t:
      if prev is None or e['acc'] == prev['acc']:
        return e['step']
      return prev['step'] + (t - prev['acc']) * (e['step'] - prev['step']) / (e['acc'] - prev['acc'])
    prev = e
  return None


def speedups(base, var, ts=EARLY):
  out = []
  for t in ts:
    a, b = (tstep(base, t), tstep(var, t)) if base and var else (None, None)
    out.append(a / b if a and b else None)
  return out


def at30(r):
  return next((e for e in r['log'] if e['acc'] >= 0.3 and 'pass_probe' in e), None)


def f(x, p=3):
  return '-' if x is None else f'{x:.{p}f}'


def pts(x):
  return '-' if x is None else f'{100 * x:+.1f}'


def rng(xs, p=2):
  xs = [x for x in xs if x is not None]
  return '-' if not xs else f'{min(xs):.{p}f}–{max(xs):.{p}f}'


def avg(xs):
  xs = [x for x in xs if x is not None]
  return mean(xs) if xs else None


def verdict(flags):
  """flags: one bool (or None: missing) per seed. holds if all True, refuted if all False, else split / pending."""
  if any(x is None for x in flags):
    return 'pending'
  return 'holds' if all(flags) else 'refuted' if not any(flags) else 'split'


def table(header, rows):
  return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(r) + ' |' for r in rows]


def get(runs, arm, K, s):
  return runs.get(f'{arm}_K{K}_s{s}')


def section_a(level, level_long):
  L, S = ['## Test A: level of capacity-limited layers (30,000 steps, cosine decay)\n'], {}
  pairs = ((32, 11), (64, 21), (128, 42))
  arms = [(f'cos_w{D}_s4_lvl', 'cos', D) for D, _ in pairs] + [(f'geglu_w{w}_lvl', 'geglu', w) for _, w in pairs]
  rows, limited = [], {}
  for arm, fam, width in arms:
    acc = {K: [late(get(level, arm, K, s)) if get(level, arm, K, s) else None for s in SEEDS] for K in (1, 4, 8)}
    lim = None if None in acc[1] else all(a < 0.97 for a in acc[1])
    limited[arm] = lim
    gains = {K: [None if None in (acc[K][i], acc[1][i]) else acc[K][i] - acc[1][i] for i in range(2)] for K in (4, 8)}
    params = next((get(level, arm, 1, s)['params'] for s in SEEDS if get(level, arm, 1, s)), None)
    S[arm] = {'late': acc, 'gain': gains, 'capacity_limited': lim, 'params': params}
    rows.append([arm, f'{params:,}' if params else '-', ' / '.join(f(a) for a in acc[1]), ' / '.join(f(a) for a in acc[4]),
                 ' / '.join(f(a) for a in acc[8]), ' / '.join(pts(g) for g in gains[4]),
                 ' / '.join(pts(g) for g in gains[8]), {True: 'yes', False: 'no', None: '-'}[lim]])
  L += table(['arm', 'trained params', 'late K = 1 (s1 / s2)', 'K = 4', 'K = 8', 'gain K = 4 (points)', 'gain K = 8',
              'capacity-limited'], rows)
  V = {}
  for D, w in pairs:
    c, g = f'cos_w{D}_s4_lvl', f'geglu_w{w}_lvl'
    if limited.get(c):
      V[f'A1 D={D}'] = verdict([None if x is None else x >= 0.01 for x in S[c]['gain'][4]])
      V[f'A4 D={D}'] = verdict([None if None in (S[c]['late'][8][i], S[c]['late'][4][i]) else
                                S[c]['late'][8][i] >= S[c]['late'][4][i] - 0.005 for i in range(2)])
      if limited.get(g):
        V[f'A2 D={D} / w={w}'] = verdict([None if None in (S[c]['gain'][4][i], S[g]['gain'][4][i]) else
                                          S[c]['gain'][4][i] - S[g]['gain'][4][i] >= 0.01 for i in range(2)])
  sel = {}
  for fam in ('cos', 'geglu'):
    cands = [(abs(mean(S[a]['late'][1]) - 0.85), a) for a, fm, _ in arms if fm == fam and limited.get(a)]
    sel[fam] = min(cands)[1] if cands else None
  S['a_long_selection'] = sel
  L.append(f'\nA-long selection (capacity-limited width with K = 1 late accuracy closest to 0.85): {sel}')
  if level_long:
    rows = []
    for fam, arm in sel.items():
      if not arm:
        continue
      la = arm.replace('_lvl', '_lvl60')
      acc = {K: [late(get(level_long, la, K, s)) if get(level_long, la, K, s) else None for s in SEEDS] for K in (1, 4)}
      g60 = [None if None in (acc[4][i], acc[1][i]) else acc[4][i] - acc[1][i] for i in range(2)]
      g30 = S[arm]['gain'][4]
      V[f'A3 {arm}'] = verdict([None if None in (g60[i], g30[i]) else (g30[i] <= 0 or g60[i] >= 0.5 * g30[i])
                                for i in range(2)])
      S[la] = {'late': acc, 'gain60': g60, 'gain30': g30}
      rows.append([la, ' / '.join(f(a) for a in acc[1]), ' / '.join(f(a) for a in acc[4]), ' / '.join(pts(g) for g in g60),
                   ' / '.join(pts(g) for g in g30)])
    L += ['\nA-long (60,000 steps)\n'] + table(['arm', 'late K = 1', 'late K = 4', 'gain at 60k', 'gain at 30k'], rows)
  rows = []
  for arm, fam, width in arms:
    for K in (4, 8):
      for s in SEEDS:
        r = get(level, arm, K, s)
        if r and 'loop_probe' in r:
          pp = r['log'][-1].get('pass_probe', {})
          rows.append([f'{arm} K = {K} s{s}', ' '.join(f(x, 2) for x in pp.get('loss', [])),
                       ' '.join(f(r['loop_probe']['acc_by_K'][k], 3) for k in ('1', '2', '4', '8', '16'))])
  if rows:
    L += ['\nPer-pass readout loss at the end (pass 0 = before the looped block) and accuracy at K_test = 1 / 2 / 4 / 8 / 16\n']
    L += table(['run', 'loss after pass 0 .. K', 'acc at K_test'], rows)
  return L, S, V


def section_b(gate, why, infiloop_old):
  L, S, V = ['\n## Test B: gate activity and deep supervision (10,000 steps)\n'], {}, {}
  rows = []
  for arm in ('cos_w1536_s4', 'geglu_w512'):
    for s in SEEDS:
      r = gate.get(f'{arm}_infiloop_K4_s{s}')
      if not r:
        continue
      lp, last = r['loop_probe'], r['log'][-1]
      dq0 = lp['acc_by_K_q0']['4'] - lp['acc_by_K']['4']
      sd = max(last['pass_probe']['gate']['logit_sd']) if 'pass_probe' in last else None
      old = infiloop_old.get(f'{arm}_infiloop_K4_s{s}')
      S[f'{arm}_infiloop_K4_s{s}'] = {'q_norm': last['gate_q_norm'], 'beta': last['gate_beta'], 'q0_change': dq0,
                                      'logit_sd_max': sd, 'late': late(r), 'late_old': late(old) if old else None}
      rows.append([f'{arm} s{s}', f(last['gate_q_norm'], 2), f(last['gate_beta'], 3), pts(dq0), f(sd, 2),
                   f(late(r)), f(late(old)) if old else '-'])
  L += table(['gated K = 4, no deep supervision', '|q|', 'β', 'gate-off change at K (points)', 'max score SD (nats)',
              'late acc', 'late acc, Test 2 run'], rows)
  for arm in ('cos_w1536_s4', 'geglu_w512'):
    xs = [S.get(f'{arm}_infiloop_K4_s{s}') for s in SEEDS]
    V[f'B1 {arm}'] = verdict([None if x is None or x['logit_sd_max'] is None else
                              abs(x['q0_change']) < 0.01 and x['logit_sd_max'] < 0.5 for x in xs])
  rows = []
  for arm in ('cos_w1536_s4', 'geglu_w512'):
    for tag in ('ds', 'infiloop_ds'):
      for K in (4, 8):
        flags, sp_flags = [], []
        for s in SEEDS:
          r, base = gate.get(f'{arm}_{tag}_K{K}_s{s}'), get(why, arm, 1, s)
          if not r:
            flags.append(None)
            sp_flags.append(None)
            continue
          a = r['loop_probe']['acc_by_K']
          drop = a[str(K)] - a[str(2 * K)]
          sp = speedups(base, r)
          q0 = (r['loop_probe']['acc_by_K_q0'][str(K)] - a[str(K)]) if 'acc_by_K_q0' in r['loop_probe'] else None
          nods = gate.get(f'{arm}_infiloop_K{K}_s{s}') if tag == 'infiloop_ds' else None
          S[f'{arm}_{tag}_K{K}_s{s}'] = {'late': late(r), 'acc_K': a[str(K)], 'acc_2K': a[str(2 * K)], 'drop': drop,
                                         'speedup_early': sp, 'q0_change': q0,
                                         'q_norm': r['log'][-1].get('gate_q_norm')}
          flags.append(drop <= 0.02)
          sp_flags.append(None if None in sp else (min(sp) >= 1.2 if arm.startswith('cos') else max(sp) <= 1.05))
          rows.append([f'{arm} {tag} K = {K} s{s}', f(late(r)), f(a[str(K)]), f(a[str(2 * K)]), pts(drop), rng(sp),
                       pts(q0) if q0 is not None else '-'])
        V[f'B2 {arm} {tag} K={K}'] = verdict(flags)
        if K == 4:
          V[f'B3 {arm} {tag}'] = verdict(sp_flags)
  L += ['\nDeep supervision\n'] + table(['run', 'late acc', 'acc at K', 'acc at 2K', 'drop K → 2K (points)',
                                         'early speed-up vs K = 1', 'gate-off change'], rows)
  cg = [S.get(f'cos_w1536_s4_infiloop_ds_K{K}_s{s}') for K in (4, 8) for s in SEEDS]
  V['B4 cos gated deep supervision'] = ('pending' if any(x is None for x in cg) else
                                        'holds' if any(x['q0_change'] is not None and abs(x['q0_change']) >= 0.01
                                                       for x in cg) else 'refuted')
  return L, S, V


SIGMAS = (('1p5', 1.5), ('2', 2), ('3', 3), ('4', 4), ('6', 6), ('8', 8))


def spearman(x, y):
  rank = lambda v: [sorted(v).index(a) + 1 for a in v]
  rx, ry = rank(x), rank(y)
  n = len(x)
  return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))


def section_c(why, width_old):
  L, S, V = ['\n## Test C: why the narrow cosine gains (10,000 steps)\n'], {}, {}
  rows, sp_mean, overlap = [], {}, {}
  for tag, sig in SIGMAS:
    arm = f'cos_w1536_s{tag}'
    sp = {s: speedups(get(why, arm, 1, s), get(why, arm, 4, s)) for s in SEEDS}
    hi = {s: speedups(get(why, arm, 1, s), get(why, arm, 4, s), HIGH) for s in SEEDS}
    sp_mean[sig] = avg(sp[1] + sp[2])
    ov = []
    for s in SEEDS:
      r = get(why, arm, 4, s)
      e = at30(r) if r else None
      ov.append(avg(e['pass_probe'].get('overlap_prev', [])) if e else None)
    overlap[sig] = avg(ov)
    lk = {K: [late(get(why, arm, K, s)) if get(why, arm, K, s) else None for s in SEEDS] for K in (1, 4)}
    S[arm] = {'speedup_early': sp, 'speedup_high': hi, 'mean_early': sp_mean[sig], 'overlap30': ov, 'late': lk}
    rows.append([f'σ = {sig}', ' / '.join(f(a) for a in lk[1]), ' / '.join(f(a) for a in lk[4]),
                 f'{rng(sp[1])} / {rng(sp[2])}', f(sp_mean[sig], 2), f'{rng(hi[1])} / {rng(hi[2])}',
                 ' / '.join(f(o, 3) for o in ov)])
  L += table(['Cos D = 1,536', 'late K = 1 (s1 / s2)', 'late K = 4', 'early speed-up K = 4, s1 / s2', 'mean',
              'high speed-up (0.7, 0.9)', 'overlap_prev at 30%'], rows)
  vals = [sp_mean[s] for _, s in SIGMAS]
  if None in vals:
    V['C1'] = 'pending'
  else:
    peak = max(range(6), key=lambda i: vals[i])
    if SIGMAS[peak][1] in (2, 3, 4) and vals[0] <= vals[peak] - 0.1 and vals[5] <= vals[peak] - 0.1:
      V['C1'] = 'peak (holds)'
    elif all(vals[i + 1] <= vals[i] + 0.05 for i in range(5)):
      V['C1'] = 'monotone (refuted)'
    else:
      V['C1'] = 'other (refuted)'
  ovs = [overlap[s] for _, s in SIGMAS]
  if None in ovs or None in vals:
    V['C4'] = 'pending'
  else:
    rho = spearman(ovs, vals)
    S['C4_spearman'] = rho
    o4, o8 = S['cos_w1536_s4']['overlap30'], S['cos_w1536_s8']['overlap30']
    V['C4'] = verdict([o4[i] < o8[i] for i in range(2)]) + f' (σ = 4 below σ = 8), Spearman {rho:+.2f} ' + \
              ('(≤ −0.6: holds)' if rho <= -0.6 else '(> −0.6: refuted)')
  if width_old:
    S['replication_sigma4'] = {s: [late(get(why, 'cos_w1536_s4', K, s)) if get(why, 'cos_w1536_s4', K, s) else None,
                                   late(get(width_old, 'cos_w1536_s4', K, s)) if get(width_old, 'cos_w1536_s4', K, s) else None]
                               for K in (1, 4) for s in SEEDS}
  # C2: output scale; C2b: GeGLU
  rows = []
  base_sp = {}
  for arm in ('cos_w1536_s4', 'cos_w1536_s8', 'geglu_w512'):
    k4 = {s: speedups(get(why, arm, 1, s), get(why, arm, 4, s)) for s in SEEDS}
    base_sp[arm] = avg(k4[1] + k4[2])
    for a in (2, 4):
      sp = {s: speedups(get(why, arm, 1, s), get(why, f'{arm}_a{a}', 1, s)) for s in SEEDS}
      S[f'{arm}_a{a}'] = {'speedup_early': sp, 'mean': avg(sp[1] + sp[2]),
                          'late': [late(get(why, f'{arm}_a{a}', 1, s)) if get(why, f'{arm}_a{a}', 1, s) else None
                                   for s in SEEDS]}
      rows.append([f'{arm}, K = 1, a = {a}', f'{rng(sp[1])} / {rng(sp[2])}', f(S[f'{arm}_a{a}']['mean'], 2),
                   f(base_sp[arm], 2), ' / '.join(f(x) for x in S[f'{arm}_a{a}']['late'])])
  L += ['\nOutput scale: K = 1 with step a, speed-up over K = 1 with a = 1\n']
  L += table(['run', 'early speed-up s1 / s2', 'mean', 'K = 4 mean (for reference)', 'late acc'], rows)
  m4, k4 = S['cos_w1536_s4_a4']['mean'], base_sp['cos_w1536_s4']
  V['C2'] = 'pending' if None in (m4, k4) else ('holds' if m4 - 1 >= 0.5 * (k4 - 1) else 'refuted') + \
            f' (a = 4: {m4:.2f}, K = 4: {k4:.2f})'
  g = S['geglu_w512_a4']['speedup_early']
  V['C2b'] = verdict([None if None in g[s] else max(g[s]) < 1 for s in SEEDS])
  # C3: learning rate
  rows = []
  for arm in ('cos_w1536_s4', 'geglu_w512'):
    variants = {('K1', 1e-3): (arm, 1), ('K1', 2e-3): (f'{arm}_lr2e3', 1), ('K1', 4e-3): (f'{arm}_lr4e3', 1),
                ('K4', 1e-3): (arm, 4), ('K4', 2e-3): (f'{arm}_lr2e3', 4)}
    steps = {}
    for (k, lr), (a, K) in variants.items():
      per = {s: [tstep(get(why, a, K, s), t) if get(why, a, K, s) else None for t in EARLY] for s in SEEDS}
      steps[(k, lr)] = per
      flat = per[1] + per[2]
      rows.append([f'{arm} {k} lr {lr:g}', ' / '.join(' '.join(f(x, 0) for x in per[s]) for s in SEEDS),
                   f(avg(flat), 0) if None not in flat else '-',
                   ' / '.join(f(late(get(why, a, K, s))) if get(why, a, K, s) else '-' for s in SEEDS)])
    ok = {key: None not in (v[1] + v[2]) for key, v in steps.items()}
    if all(ok.values()):
      b1 = min((avg(steps[('K1', lr)][1] + steps[('K1', lr)][2]), lr) for lr in (1e-3, 2e-3, 4e-3))[1]
      b4 = min((avg(steps[('K4', lr)][1] + steps[('K4', lr)][2]), lr) for lr in (1e-3, 2e-3))[1]
      flags = [all(steps[('K1', b1)][s][i] / steps[('K4', b4)][s][i] >= 1.15 for i in range(len(EARLY))) for s in SEEDS]
      S[f'C3 {arm}'] = {'best_K1_lr': b1, 'best_K4_lr': b4,
                        'ratio': {s: [steps[('K1', b1)][s][i] / steps[('K4', b4)][s][i] for i in range(len(EARLY))]
                                  for s in SEEDS}}
      if arm.startswith('cos'):
        V['C3'] = verdict(flags) + f' (best K = 1 lr {b1:g}, best K = 4 lr {b4:g})'
      else:
        V['C3 (GeGLU, no rule)'] = verdict(flags) + f' (best K = 1 lr {b1:g}, best K = 4 lr {b4:g})'
    elif arm.startswith('cos'):
      V['C3'] = 'pending (a run does not reach every early threshold)' if steps else 'pending'
  L += ['\nLearning rate: step to 20 / 30 / 40 / 50% accuracy, s1 / s2\n']
  L += table(['run', 'steps s1 / s2', 'mean', 'late acc s1 / s2'], rows)
  # C5 and the probe at the 30% point
  rows = []
  desc = {}
  for arm in ('cos_w1536_s4', 'cos_w1536_s8', 'cos_w1536_s1p5', 'geglu_w512', 'rfrelu_w1536'):
    for s in SEEDS:
      r = get(why, arm, 4, s)
      e = at30(r) if r else None
      if not e:
        continue
      p = e['pass_probe']
      desc[(arm, s)] = p['descent_cos']
      rows.append([f'{arm} K = 4 s{s}', str(e['step']), ' '.join(f(x, 2) for x in p['loss']),
                   ' '.join(f(x, 2) for x in p['descent_cos']), ' '.join(f(x, 3) for x in p['rel_step']),
                   ' '.join(f(x, 2) for x in p.get('input_shift', [])), ' '.join(f(x, 3) for x in p.get('overlap_prev', []))])
  L += ['\nPer-pass probe at the 30% point of each K = 4 run\n']
  L += table(['run', 'step', 'loss after pass 0 .. 4', 'descent_cos', 'rel_step', 'input_shift', 'overlap_prev'], rows)
  fl = []
  for s in SEEDS:
    c, g = desc.get(('cos_w1536_s4', s)), desc.get(('geglu_w512', s))
    fl.append(None if c is None or g is None else (min(c) > 0 and mean(c) > mean(g)))
  V['C5'] = verdict(fl)
  S['descent_cos_30'] = {f'{a}_s{s}': v for (a, s), v in desc.items()}
  return L, S, V


def main():
  p = argparse.ArgumentParser()
  for k in ('level', 'level_long', 'gate', 'why', 'width_old', 'infiloop_old', 'out'):
    p.add_argument(f'--{k}', default=None)
  args = p.parse_args()
  level, why, gate = load(args.level), load(args.why), load(args.gate)
  LA, SA, VA = section_a(level, load(args.level_long) if args.level_long else None)
  LB, SB, VB = section_b(gate, why, load(args.infiloop_old))
  LC, SC, VC = section_c(why, load(args.width_old) if args.width_old else None)
  V = VA | VB | VC
  lines = LA + LB + LC + ['\n## Verdicts\n'] + table(['prediction', 'verdict'], [[k, v] for k, v in V.items()])
  print('\n'.join(lines))
  if args.out:
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, 'tables.md'), 'w').write('\n'.join(lines) + '\n')
    json.dump({'A': SA, 'B': SB, 'C': SC, 'verdicts': V}, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1,
              default=str)


if __name__ == '__main__':
  main()
