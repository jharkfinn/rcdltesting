"""Tables and verdicts for tests A, B and C of the looping follow-ups and their amendments
(docs/data/loops/predictions_abc.md, predictions_abc_amendment_1.md, predictions_abc_amendment_2.md).

Usage: python -m experiments.loops_abc_analysis --level DIR --gate DIR --why DIR [--grid DIR] [--level2 DIR]
       [--width_old DIR] [--infiloop_old DIR] [--out DIR]
Each DIR holds <name>/result.json; names end in _K{K}_s{seed}. Missing runs are reported as '-' and the rules that need
them as 'pending'. With --out, writes tables.md and summary.json (every number the tables and verdicts use).

Measures (as defined in the predictions):
  late accuracy: mean over the evaluations at steps >= 0.75 * steps;
  final accuracy (cosine-decayed runs of amendment 2): mean of the last 3 evaluations;
  gain: accuracy of a variant minus that of its baseline, same seed, in points;
  speed-up at threshold t: (step at which the baseline first reaches t) / (step at which the variant first reaches t),
    steps linearly interpolated between evaluations; early thresholds 0.2-0.5, high thresholds 0.7 and 0.9;
  thresholds used (amendment 2): the early thresholds that every run in a comparison reaches; fewer than 2 -> 'not
    testable'; a run with late accuracy < 0.1 did not train and is never 'best';
  every rule per seed: holds if both seeds pass, refuted if both fail, split otherwise.
"""

import argparse
import glob
import json
import os
from statistics import mean

EARLY, HIGH = (0.2, 0.3, 0.4, 0.5), (0.7, 0.9)
SEEDS = (1, 2)


def load(*roots):
  runs = {}
  for root in roots:
    for f in sorted(glob.glob(os.path.join(root, '*', 'result.json'))) if root else []:
      runs[os.path.basename(os.path.dirname(f))] = json.load(open(f))
  return runs


def late(r, key='acc', frac=0.75):
  if not r:
    return None
  v = [e[key] for e in r['log'] if frac * r['cfg']['steps'] <= e['step'] <= r['cfg']['steps'] and key in e]
  return mean(v) if v else None


def final(r, n=3):
  if not r:
    return None
  v = [e['acc'] for e in r['log'] if e['step'] <= r['cfg']['steps']]
  return mean(v[-n:])


def trained(r):
  return r is not None and late(r) >= 0.1


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


def used(runs, ts=EARLY):
  """Thresholds reached by every run in the comparison (all must exist)."""
  if any(r is None for r in runs):
    return None
  return [t for t in ts if all(tstep(r, t) is not None for r in runs)]


def ratios(base, var, ts):
  return [tstep(base, t) / tstep(var, t) for t in ts]


def speedups(base, var, ts=EARLY):
  out = []
  for t in ts:
    a, b = (tstep(base, t), tstep(var, t)) if base and var else (None, None)
    out.append(a / b if a and b else None)
  return out


def at_acc(r, thr):
  return next((e for e in r['log'] if e['acc'] >= thr and 'pass_probe' in e), None) if r else None


def at_step(r, step):
  return next((e for e in r['log'] if e['step'] == step and 'pass_probe' in e), None) if r else None


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
  """flags: one value per seed: True / False, None (missing run) or 'nt' (not testable)."""
  if any(x is None for x in flags):
    return 'pending'
  if any(x == 'nt' for x in flags):
    return 'not testable'
  return 'holds' if all(flags) else 'refuted' if not any(flags) else 'split'


def table(header, rows):
  return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(r) + ' |' for r in rows]


def get(runs, arm, K, s):
  return runs.get(f'{arm}_K{K}_s{s}')


def atag(a):
  return '' if a == 1 else '_a' + f'{a:g}'.replace('.', 'p')


# ---------------------------------------------------------------- Test A (and A-long)

def section_a(level):
  L, S = ['## Test A: level of capacity-limited layers (30,000 steps, cosine decay)\n'], {}
  pairs = ((32, 11), (64, 21), (128, 42))
  arms = [(f'cos_w{D}_s4_lvl', 'cos', D) for D, _ in pairs] + [(f'geglu_w{w}_lvl', 'geglu', w) for _, w in pairs]
  rows, limited = [], {}
  for arm, fam, width in arms:
    acc = {K: [late(get(level, arm, K, s)) for s in SEEDS] for K in (1, 4, 8)}
    sc = {k: [late(get(level, f'{arm}{atag(a)}', K, s)) for s in SEEDS] for k, (a, K) in
          {'K1_a4': (4, 1), 'K4_a0.25': (0.25, 4)}.items()}
    lim = None if None in acc[1] else all(a < 0.97 for a in acc[1])
    limited[arm] = lim
    gains = {K: [None if None in (acc[K][i], acc[1][i]) else acc[K][i] - acc[1][i] for i in range(2)] for K in (4, 8)}
    params = next((get(level, arm, 1, s)['params'] for s in SEEDS if get(level, arm, 1, s)), None)
    S[arm] = {'late': acc, 'late_scale': sc, 'gain': gains, 'capacity_limited': lim, 'params': params}
    rows.append([arm, f'{params:,}' if params else '-', ' / '.join(f(a) for a in acc[1]), ' / '.join(f(a) for a in acc[4]),
                 ' / '.join(f(a) for a in acc[8]), ' / '.join(f(a) for a in sc['K1_a4']),
                 ' / '.join(f(a) for a in sc['K4_a0.25']), ' / '.join(pts(g) for g in gains[4]),
                 {True: 'yes', False: 'no', None: '-'}[lim]])
  L += table(['arm', 'trained params', 'late K = 1 (s1 / s2)', 'K = 4', 'K = 8', 'K = 1, a = 4', 'K = 4, a = 0.25',
              'gain K = 4 (points)', 'capacity-limited'], rows)
  V = {}
  cos_lim = [D for D, _ in pairs if limited.get(f'cos_w{D}_s4_lvl')]
  if all(limited.get(f'cos_w{D}_s4_lvl') is not None for D, _ in pairs) and not cos_lim:
    for k in ('A1', 'A2', 'A4', 'D3'):
      V[k] = 'not testable (no capacity-limited Cos width)'
  for D, w in pairs:
    c, g = f'cos_w{D}_s4_lvl', f'geglu_w{w}_lvl'
    if limited.get(c):
      V[f'A1 D={D}'] = verdict([None if x is None else x >= 0.01 for x in S[c]['gain'][4]])
      V[f'A4 D={D}'] = verdict([None if None in (S[c]['late'][8][i], S[c]['late'][4][i]) else
                                S[c]['late'][8][i] >= S[c]['late'][4][i] - 0.005 for i in range(2)])
      if limited.get(g):
        V[f'A2 D={D} / w={w}'] = verdict([None if None in (S[c]['gain'][4][i], S[g]['gain'][4][i]) else
                                          S[c]['gain'][4][i] - S[g]['gain'][4][i] >= 0.01 for i in range(2)])
  for arm, fam, width in arms:                    # D4 (amendment 1) for capacity-limited GeGLU widths
    if fam == 'geglu' and limited.get(arm):
      d1 = [None if None in (S[arm]['late_scale']['K4_a0.25'][i], S[arm]['late'][1][i]) else
            S[arm]['late_scale']['K4_a0.25'][i] - S[arm]['late'][1][i] for i in range(2)]
      d2 = [None if None in (S[arm]['late'][4][i], S[arm]['late_scale']['K1_a4'][i]) else
            S[arm]['late'][4][i] - S[arm]['late_scale']['K1_a4'][i] for i in range(2)]
      S[arm]['D4_differences'] = {'K4a0.25-K1a1': d1, 'K4a1-K1a4': d2}
      V[f'D4 {arm}'] = verdict([None if None in (d1[i], d2[i]) else (d1[i] < 0.01 and d2[i] < 0.01) for i in range(2)])
  sel = {}
  for fam in ('cos', 'geglu'):
    cands = [(abs(mean(S[a]['late'][1]) - 0.85), a) for a, fm, _ in arms if fm == fam and limited.get(a)]
    sel[fam] = min(cands)[1] if cands else None
  S['a_long_selection'] = sel
  L.append(f'\nA-long selection (capacity-limited width with K = 1 late accuracy closest to 0.85): {sel}')
  rows = []
  for fam, arm in sel.items():
    if not arm:
      continue
    l60, l120 = arm.replace('_lvl', '_lvl60'), arm.replace('_lvl', '_lvl120')
    k4_30 = [final(get(level, arm, 4, s)) for s in SEEDS]
    k1_30 = [final(get(level, arm, 1, s)) for s in SEEDS]
    k1_60, k4_60 = [final(get(level, l60, 1, s)) for s in SEEDS], [final(get(level, l60, 4, s)) for s in SEEDS]
    k1_120 = [final(get(level, l120, 1, s)) for s in SEEDS]
    g30 = [None if None in (k4_30[i], k1_30[i]) else k4_30[i] - k1_30[i] for i in range(2)]
    g60 = [None if None in (k4_60[i], k1_60[i]) else k4_60[i] - k1_60[i] for i in range(2)]
    strict = [None if None in (k4_30[i], k1_120[i]) else k4_30[i] - k1_120[i] for i in range(2)]
    S[f'{arm}_long'] = {'final_K1_30k': k1_30, 'final_K4_30k': k4_30, 'final_K1_60k': k1_60, 'final_K4_60k': k4_60,
                        'final_K1_120k': k1_120, 'gain30': g30, 'gain60': g60, 'K4_30k_minus_K1_120k': strict}
    V[f'A3 {arm} (as written)'] = verdict([None if None in (g30[i], g60[i]) else
                                           ('nt' if g30[i] < 0.01 else g60[i] >= 0.5 * g30[i]) for i in range(2)])
    V[f'A-long strict {arm}'] = verdict([None if x is None else x >= 0.01 for x in strict])
    rows.append([arm, ' / '.join(f(x) for x in k1_30), ' / '.join(f(x) for x in k4_30), ' / '.join(f(x) for x in k1_60),
                 ' / '.join(f(x) for x in k4_60), ' / '.join(f(x) for x in k1_120), ' / '.join(pts(x) for x in strict)])
  L += ['\nA-long (final accuracy = mean of the last 3 evaluations)\n']
  L += table(['arm', 'K = 1, 30k', 'K = 4, 30k', 'K = 1, 60k', 'K = 4, 60k', 'K = 1, 120k', 'K4(30k) − K1(120k)'], rows)
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


# ---------------------------------------------------------------- Test B

def section_b(gate, why, infiloop_old):
  L, S, V = ['\n## Test B: gate activity and deep supervision (10,000 steps)\n'], {}, {}
  rows = []
  for arm in ('cos_w1536_s4', 'geglu_w512'):
    flags = []
    for s in SEEDS:
      r = gate.get(f'{arm}_infiloop_K4_s{s}')
      if not r:
        flags.append(None)
        continue
      lp, last = r['loop_probe'], r['log'][-1]
      dq0 = lp['acc_by_K_q0']['4'] - lp['acc_by_K']['4']
      g = last['pass_probe']['gate']
      wsd, lsd = max(g['weight_sd']), max(g['logit_sd'])
      old = infiloop_old.get(f'{arm}_infiloop_K4_s{s}')
      S[f'{arm}_infiloop_K4_s{s}'] = {'q_norm': last['gate_q_norm'], 'beta': last['gate_beta'], 'q0_change': dq0,
                                      'weight_sd_max': wsd, 'logit_sd_max': lsd, 'weight_median': g['weight_median'],
                                      'late': late(r), 'late_old': late(old)}
      flags.append(abs(dq0) < 0.01 and wsd < 0.02)
      rows.append([f'{arm} s{s}', f(last['gate_q_norm'], 2), f(last['gate_beta'], 3), pts(dq0), f(wsd, 3), f(lsd, 2),
                   ' '.join(f(x, 2) for x in g['weight_median']), f(late(r)), f(late(old))])
    V[f'B1 {arm}'] = verdict(flags)
  L += table(['gated K = 4, no deep supervision', '|q|', 'β', 'gate-off change at K (points)', 'max weight SD',
              'max score SD (nats)', 'median weights z_0 .. z_4', 'late acc', 'late acc, Test 2 run'], rows)
  rows, b3 = [], []
  for arm in ('cos_w1536_s4', 'geglu_w512'):
    for tag in ('ds', 'infiloop_ds'):
      nods_tag = '' if tag == 'ds' else '_infiloop'
      for K in (4, 8):
        flags, sp_flags = [], []
        for s in SEEDS:
          r, base = gate.get(f'{arm}_{tag}_K{K}_s{s}'), get(why, arm, 1, s)
          nods = (why if (K == 4 and tag == 'ds') else gate).get(f'{arm}{nods_tag}_K{K}_s{s}')
          if not r:
            flags.append(None)
            sp_flags.append(None)
            continue
          a = r['loop_probe']['acc_by_K']
          drop = a[str(K)] - a[str(2 * K)]
          works = a[str(K)] - a['1'] >= 0.02
          drop_nods = None
          if nods:
            an = nods['loop_probe']['acc_by_K']
            drop_nods = an[str(K)] - an[str(2 * K)]
          q0 = (r['loop_probe']['acc_by_K_q0'][str(K)] - a[str(K)]) if 'acc_by_K_q0' in r['loop_probe'] else None
          ts = used([base, r]) if base else None
          sp = ratios(base, r, ts) if ts and len(ts) >= 2 else None
          sp_ds = None
          if nods and K == 4:
            tsn = used([nods, r])
            sp_ds = ratios(nods, r, tsn) if tsn and len(tsn) >= 2 else None
          S[f'{arm}_{tag}_K{K}_s{s}'] = {'late': late(r), 'late_no_ds': late(nods), 'acc_by_K': a, 'drop': drop,
                                         'drop_no_ds': drop_nods, 'loop_works': works, 'speedup_vs_K1': sp,
                                         'thresholds_used': ts, 'speedup_vs_same_K_no_ds': sp_ds, 'q0_change': q0,
                                         'q_norm': r['log'][-1].get('gate_q_norm')}
          flags.append(('nt' if not works else drop <= 0.02))
          if K == 4:
            if base is None:
              sp_flags.append(None)
            elif sp is None:
              sp_flags.append('nt')
            else:
              sp_flags.append(min(sp) >= 1.2 if arm.startswith('cos') else max(sp) <= 1.05)
          rows.append([f'{arm} {tag} K = {K} s{s}', f(late(r)), f(late(nods)), f(a['1']), f(a[str(K)]), f(a[str(2 * K)]),
                       pts(drop), pts(drop_nods), rng(sp or []), rng(sp_ds or []), pts(q0) if q0 is not None else '-'])
        V[f'B2 {arm} {tag} K={K}'] = verdict(flags).replace('not testable', 'loop idle')
        if K == 4:
          b3.append(verdict(sp_flags))
  L += ['\nDeep supervision (drop = accuracy at K minus accuracy at 2K)\n'] + table(
    ['run', 'late acc', 'late acc, no deep sup.', 'acc at 1 pass', 'acc at K', 'acc at 2K', 'drop (points)',
     'drop, no deep sup.', 'speed-up vs K = 1', 'speed-up vs same K, no deep sup.', 'gate-off change'], rows)
  V['B3 (all four arms)'] = ('pending' if 'pending' in b3 else 'not testable' if 'not testable' in b3 else
                             'holds' if all(v == 'holds' for v in b3) else 'refuted' if all(v == 'refuted' for v in b3)
                             else f'not met ({", ".join(b3)})')
  flags = []
  for s in SEEDS:
    x = S.get(f'cos_w1536_s4_infiloop_ds_K4_s{s}')
    flags.append(None if x is None or x['q0_change'] is None else abs(x['q0_change']) >= 0.01)
  V['B4 cos gated deep supervision, K = 4'] = verdict(flags)
  k8 = [S.get(f'cos_w1536_s4_infiloop_ds_K8_s{s}') for s in SEEDS]
  V['B4 K = 8 (reported)'] = ' / '.join('-' if x is None else pts(x['q0_change']) for x in k8)
  return L, S, V


# ---------------------------------------------------------------- Test C (and amendment 1)

SIGMAS = (('1p5', 1.5), ('2', 2), ('3', 3), ('4', 4), ('6', 6), ('8', 8))


def spearman(x, y):
  rank = lambda v: [sorted(v).index(a) + 1 for a in v]
  rx, ry = rank(x), rank(y)
  n = len(x)
  return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))


def classify(vals):
  """C1 order: peak, else monotone, else other (vals: mean speed-up per sigma in SIGMAS order)."""
  peak = max(range(len(vals)), key=lambda i: vals[i])
  if SIGMAS[peak][1] in (2, 3, 4) and vals[0] <= vals[peak] - 0.1 and vals[-1] <= vals[peak] - 0.1:
    return 'peak'
  if all(vals[i + 1] <= vals[i] + 0.05 for i in range(len(vals) - 1)):
    return 'monotone'
  return 'other'


def section_c(why, width_old):
  L, S, V = ['\n## Test C: why the narrow cosine gains (10,000 steps)\n'], {}, {}
  rows, per_seed_mean, overlap = [], {s: {} for s in SEEDS}, {}
  for tag, sig in SIGMAS:
    arm = f'cos_w1536_s{tag}'
    sp, hi, ov, ts_all = {}, {}, {}, {}
    for s in SEEDS:
      b, v = get(why, arm, 1, s), get(why, arm, 4, s)
      ts = used([b, v])
      ts_all[s] = ts
      sp[s] = ratios(b, v, ts) if ts and len(ts) >= 2 else None
      hi[s] = speedups(b, v, HIGH) if b and v else [None, None]
      per_seed_mean[s][sig] = avg(sp[s]) if sp[s] else None
      ov[s] = {}
      for key, e in (('step1000', at_step(v, 1000)), ('step2000', at_step(v, 2000)), ('acc20', at_acc(v, 0.2)),
                     ('acc30', at_acc(v, 0.3))):
        if e:
          p = e['pass_probe']
          ov[s][key] = {'overlap_prev': avg(p.get('overlap_prev', [])),
                        'shift_over_sigma': avg([x / sig for x in p.get('input_shift', [])]), 'step': e['step']}
    overlap[sig] = avg([ov[s].get('acc30', {}).get('overlap_prev') for s in SEEDS])
    lk = {K: [late(get(why, arm, K, s)) for s in SEEDS] for K in (1, 4)}
    S[arm] = {'speedup_early': sp, 'thresholds_used': ts_all, 'speedup_high': hi, 'probe': ov, 'late': lk}
    rows.append([f'σ = {sig}', ' / '.join(f(a) for a in lk[1]), ' / '.join(f(a) for a in lk[4]),
                 f'{rng(sp[1] or [])} / {rng(sp[2] or [])}', f'{f(per_seed_mean[1][sig], 2)} / {f(per_seed_mean[2][sig], 2)}',
                 f'{rng(hi[1])} / {rng(hi[2])}',
                 ' / '.join(f(ov[s].get('acc30', {}).get('overlap_prev'), 3) for s in SEEDS),
                 ' / '.join(f(ov[s].get('step1000', {}).get('shift_over_sigma'), 3) for s in SEEDS)])
  L += table(['Cos D = 1,536', 'late K = 1 (s1 / s2)', 'late K = 4', 'early speed-up K = 4, s1 / s2', 'mean s1 / s2',
              'high speed-up (0.7, 0.9)', 'overlap_prev at 30%', 'input shift / σ at step 1,000'], rows)
  cls = []
  for s in SEEDS:
    vals = [per_seed_mean[s][sig] for _, sig in SIGMAS]
    missing = [sig for (tag, sig), v in zip(SIGMAS, vals) if v is None]
    if not missing:
      cls.append(classify(vals))
    elif all(get(why, f'cos_w1536_s{tag}', K, s) for tag, sig in SIGMAS if sig in missing for K in (1, 4)):
      cls.append('nt')                             # the runs exist but share fewer than 2 thresholds
    else:
      cls.append(None)
  S['C1_classes'] = cls
  V['C1 (peak predicted)'] = ('pending' if None in cls else 'not testable (a sigma shares fewer than 2 thresholds)'
                              if 'nt' in cls else f'{verdict([c == "peak" for c in cls])} (s1 {cls[0]}, s2 {cls[1]})')
  ovs = [overlap[s] for _, s in SIGMAS]
  vals = [avg([per_seed_mean[s][sig] for s in SEEDS]) for _, sig in SIGMAS]
  if None not in ovs and None not in vals:
    S['C4_spearman_descriptive'] = spearman(ovs, vals)
    V['C4 (descriptive)'] = f'Spearman(overlap at 30%, mean speed-up) = {spearman(ovs, vals):+.2f}'
  if width_old:
    S['replication_sigma4'] = {f'K{K}_s{s}': [late(get(why, 'cos_w1536_s4', K, s)), late(get(width_old, 'cos_w1536_s4', K, s))]
                               for K in (1, 4) for s in SEEDS}
  # C2 / C2b and amendment 1 (D1, D2): output scale
  rows = []
  for arm in ('cos_w1536_s4', 'cos_w1536_s8', 'geglu_w512', 'rfrelu_w1536'):
    for K, a in ((1, 2), (1, 4), (4, 0.25), (4, 1)):
      sp = {}
      for s in SEEDS:
        b, v = get(why, arm, 1, s), get(why, f'{arm}{atag(a)}', K, s)
        ts = used([b, v])
        sp[s] = ratios(b, v, ts) if ts and len(ts) >= 2 else None
      S[f'{arm}{atag(a)}_K{K}'] = {'speedup_early': sp, 'late': [late(get(why, f'{arm}{atag(a)}', K, s)) for s in SEEDS]}
      if any(get(why, f'{arm}{atag(a)}', K, s) for s in SEEDS):
        rows.append([f'{arm}, K = {K}, a = {a:g}', f'{rng(sp[1] or [])} / {rng(sp[2] or [])}',
                     ' / '.join(f(avg(sp[s] or []), 2) for s in SEEDS),
                     ' / '.join(f(x) for x in S[f'{arm}{atag(a)}_K{K}']['late'])])
  L += ['\nOutput scale: speed-up over K = 1 with a = 1 (thresholds reached by both runs)\n']
  L += table(['run', 'early speed-up s1 / s2', 'mean s1 / s2', 'late acc s1 / s2'], rows)
  c2 = []
  for s in SEEDS:
    m4, k4 = (avg(S[f'cos_w1536_s4_a4_K1']['speedup_early'][s] or []), avg(S['cos_w1536_s4_K4']['speedup_early'][s] or []))
    c2.append(None if None in (m4, k4) else (m4 - 1 >= 0.5 * (k4 - 1)))
  V['C2'] = verdict(c2)
  g = S['geglu_w512_a4_K1']['speedup_early']
  V['C2b'] = verdict([None if g[s] is None else max(g[s]) < 1 for s in SEEDS])
  d1 = S['cos_w1536_s4_a0p25_K4']['speedup_early']
  V['D1'] = verdict([None if d1[s] is None else max(d1[s]) <= 1.10 for s in SEEDS])
  d2 = S['geglu_w512_a0p25_K4']['speedup_early']
  V['D2'] = verdict([None if d2[s] is None else min(d2[s]) >= 0.95 for s in SEEDS])
  # C3: learning rate (unstable = not best; secondary to the grid of amendment 2)
  rows = []
  for arm in ('cos_w1536_s4', 'geglu_w512'):
    variants = {('K1', 1e-3): (arm, 1), ('K1', 2e-3): (f'{arm}_lr2e3', 1), ('K1', 4e-3): (f'{arm}_lr4e3', 1),
                ('K4', 1e-3): (arm, 4), ('K4', 2e-3): (f'{arm}_lr2e3', 4)}
    rr = {key: {s: get(why, a, K, s) for s in SEEDS} for key, (a, K) in variants.items()}
    for key, per in rr.items():
      rows.append([f'{arm} {key[0]} lr {key[1]:g}', ' / '.join(f(late(per[s])) for s in SEEDS),
                   ' / '.join(' '.join(f(tstep(per[s], t), 0) if per[s] and tstep(per[s], t) else '-' for t in EARLY)
                              for s in SEEDS)])
    if any(r is None for per in rr.values() for r in per.values()):
      V[f'C3 {arm}'] = 'pending'
      continue
    ok = {key: all(trained(per[s]) for s in SEEDS) for key, per in rr.items()}
    cands = [r for key, per in rr.items() if ok[key] for r in per.values()]
    ts = used(cands)
    if not ts or len(ts) < 2:
      V[f'C3 {arm}'] = 'not testable'
      continue
    msteps = lambda key: mean(tstep(rr[key][s], t) for s in SEEDS for t in ts)
    b1 = min((k for k in rr if k[0] == 'K1' and ok[k]), key=msteps)
    b4 = min((k for k in rr if k[0] == 'K4' and ok[k]), key=msteps)
    flags = [all(r >= 1.15 for r in ratios(rr[b1][s], rr[b4][s], ts)) for s in SEEDS]
    S[f'C3 {arm}'] = {'best_K1': b1, 'best_K4': b4, 'thresholds': ts,
                      'ratio': {s: ratios(rr[b1][s], rr[b4][s], ts) for s in SEEDS}}
    V[f'C3 {arm}' + ('' if arm.startswith('cos') else ' (no rule)')] = verdict(flags) + f' (best K = 1 lr {b1[1]:g}, K = 4 lr {b4[1]:g})'
  L += ['\nLearning rate (all weights): late accuracy and step to 20 / 30 / 40 / 50%\n']
  L += table(['run', 'late acc s1 / s2', 'steps s1 / s2'], rows)
  # C5 and the probe at the 30% point
  rows, desc = [], {}
  for arm in ('cos_w1536_s4', 'cos_w1536_s8', 'cos_w1536_s1p5', 'geglu_w512', 'rfrelu_w1536'):
    for K in (1, 4):
      for s in SEEDS:
        e = at_acc(get(why, arm, K, s), 0.3)
        if not e:
          continue
        p = e['pass_probe']
        desc[(arm, K, s)] = p['descent_cos']
        rows.append([f'{arm} K = {K} s{s}', str(e['step']), ' '.join(f(x, 2) for x in p['loss']),
                     ' '.join(f(x, 2) for x in p['descent_cos']), ' '.join(f(x, 3) for x in p['rel_step']),
                     ' '.join(f(x, 2) for x in p.get('input_shift', [])), ' '.join(f(x, 3) for x in p.get('overlap_prev', []))])
  L += ['\nPer-pass probe at the 30% point\n']
  L += table(['run', 'step', 'loss after pass 0 .. K', 'descent_cos', 'rel_step', 'input_shift', 'overlap_prev'], rows)
  fl = []
  for s in SEEDS:
    c, g = desc.get(('cos_w1536_s4', 4, s)), desc.get(('geglu_w512', 4, s))
    fl.append(None if c is None or g is None else (min(c) > 0 and mean(c) > mean(g)))
  V['C5'] = verdict(fl)
  S['descent_cos_30'] = {f'{a}_K{K}_s{s}': v for (a, K, s), v in desc.items()}
  return L, S, V


# ---------------------------------------------------------------- amendment 2: readout-scale grid (H1-H3)

FAMILIES = ('cos_w1536_s4', 'cos_w1536_s8', 'rfrelu_w1536', 'geglu_w512_zinit', 'relu_w768_zinit')
SCALES = (0.5, 1, 2, 4, 8, 16, 32)                # 0.5 and 32: grid extensions at the edges (amendment 2)


def section_h(runs):
  L, S, V = ['\n## Amendment 2, section 1: each family at its best readout scale (10,000 steps)\n'], {}, {}
  best, rows = {}, []
  for fam in FAMILIES:
    grid = {(K, sc): {s: get(runs, f'{fam}{atag(sc / K)}', K, s) for s in SEEDS} for K in (1, 4) for sc in SCALES}
    for ext in (0.5, 32):                            # grid extensions exist only for some families
      if all(per[s] is None for k, per in grid.items() if k[1] == ext for s in SEEDS):
        grid = {k: v for k, v in grid.items() if k[1] != ext}
    for (K, sc), per in grid.items():
      rows.append([fam, str(K), f'{sc / K:g}', str(sc), ' / '.join(f(late(per[s])) for s in SEEDS),
                   ' / '.join(' '.join(f(tstep(per[s], t), 0) if per[s] and tstep(per[s], t) else '-' for t in EARLY)
                              for s in SEEDS)])
    if any(r is None for per in grid.values() for r in per.values()):
      S[fam] = 'pending'
      continue
    ok = {k: all(trained(per[s]) for s in SEEDS) for k, per in grid.items()}
    ts = used([r for k, per in grid.items() if ok[k] for r in per.values()])
    if not ts or len(ts) < 2:
      S[fam] = {'thresholds': ts, 'verdict': 'not testable'}
      continue
    msteps = lambda k: mean(tstep(grid[k][s], t) for s in SEEDS for t in ts)
    b = {K: min((k for k in grid if k[0] == K and ok[k]), key=msteps) for K in (1, 4)}
    P = {s: ratios(grid[b[1]][s], grid[b[4]][s], ts) for s in SEEDS}
    edge = [K for K in (1, 4) if b[K][1] in (min(k[1] for k in grid), max(k[1] for k in grid))]
    best[fam] = {'grid': grid, 'best': b, 'P': P, 'thresholds': ts}
    S[fam] = {'best_scale_K1': b[1][1], 'best_scale_K4': b[4][1], 'thresholds': ts, 'P': P,
              'P_mean': {s: mean(P[s]) for s in SEEDS}, 'edge_of_grid': edge,
              'mean_steps': {f'K{k[0]}_s{k[1]}': msteps(k) for k in grid if ok[k]}}
  L += table(['family', 'K', 'a', 'scale K a', 'late acc s1 / s2', 'steps to 20/30/40/50%, s1 / s2'], rows)
  rows = []
  for fam in FAMILIES:
    x = S.get(fam)
    if isinstance(x, dict) and 'P' in x:
      rows.append([fam, f'{x["best_scale_K1"]:g}', f'{x["best_scale_K4"]:g}', ' '.join(f'{t:g}' for t in x['thresholds']),
                   ' / '.join(rng(x['P'][s]) for s in SEEDS), ' / '.join(f(x['P_mean'][s], 2) for s in SEEDS),
                   ', '.join(f'K = {k}' for k in x['edge_of_grid']) or '-'])
  L += ['\nPass gain P = step(best K = 1) / step(best K = 4)\n']
  L += table(['family', 'best scale K = 1', 'best scale K = 4', 'thresholds used', 'P range s1 / s2', 'P mean s1 / s2',
              'best at grid edge'], rows)
  c4 = best.get('cos_w1536_s4')
  if c4 is None:
    V['H1'] = V['H2'] = V['H3'] = 'pending'
  else:
    edge_note = ' (best at grid edge: extend the grid before the verdict)' if S['cos_w1536_s4']['edge_of_grid'] else ''
    V['H1'] = verdict([all(p >= 1.15 for p in c4['P'][s]) for s in SEEDS]) + edge_note
    if V['H1'].startswith('holds'):
      others = [fam for fam in FAMILIES[1:]]
      if all(fam in best for fam in others):
        V['H2'] = verdict([all(mean(c4['P'][s]) - mean(best[fam]['P'][s]) >= 0.10 for fam in others) for s in SEEDS])
      else:
        V['H2'] = 'pending'
    else:
      V['H2'] = 'refuted (H1 does not hold)' if V['H1'].startswith(('refuted', 'split')) else 'pending'
    zg = best.get('geglu_w512_zinit')
    if zg is None:
      V['H3'] = 'pending'
    else:
      cos_best4, zg_best1 = c4['grid'][c4['best'][4]], zg['grid'][zg['best'][1]]
      flags = []
      for s in SEEDS:
        ts = used([cos_best4[s], zg_best1[s]])
        flags.append('nt' if not ts or len(ts) < 2 else all(r >= 1.15 for r in ratios(zg_best1[s], cos_best4[s], ts)))
        S[f'H3_ratio_s{s}'] = ratios(zg_best1[s], cos_best4[s], ts) if ts else None
      V['H3'] = verdict(flags)
  rows = []
  for a in (1, 4):
    for s in SEEDS:
      r = get(runs, f'cos_w6144_s4{atag(a)}', 1, s)
      if r:
        rows.append([f'Cos σ = 4, D = 6,144, K = 1, a = {a} s{s}', f(late(r)),
                     ' '.join(f(tstep(r, t), 0) if tstep(r, t) else '-' for t in EARLY)])
  if rows:
    L += ['\nCompute-matched unlooped reference\n'] + table(['run', 'late acc', 'steps to 20/30/40/50%'], rows)
  return L, S, V


# ---------------------------------------------------------------- amendment 2: level test L

def section_l(runs):
  L, S, V = ['\n## Amendment 2, section 2: level test with N = 32,768 (final accuracy = mean of the last 3 evaluations)\n'], {}, {}
  rows, margin = [], {}
  for arm in ('cos_w256_s4_N32k', 'geglu_w85_zinit_N32k'):
    g = lambda name, K: [final(get(runs, name, K, s)) for s in SEEDS]
    v = {'K1 a1 60k': g(arm, 1), 'K1 a4 60k': g(f'{arm}_a4', 1), 'K1 a1 120k': g(f'{arm}_120k', 1),
         'K1 a4 120k': g(f'{arm}_a4_120k', 1), 'K4 a0.25 60k': g(f'{arm}_a0p25', 4), 'K4 a1 60k': g(arm, 4),
         'K1 a0 60k': g(f'{arm}_a0', 1)}
    S[arm] = v
    rows.append([arm] + [' / '.join(f(x) for x in v[k]) for k in v])
    pick = lambda i, keys: None if any(v[k][i] is None for k in keys) else max(v[k][i] for k in keys)
    l0 = [None if None in (v['K1 a1 60k'][i], v['K1 a0 60k'][i]) else v['K1 a1 60k'][i] - v['K1 a0 60k'][i] >= 0.03
          for i in range(2)]
    cap = [None if v['K1 a1 120k'][i] is None else v['K1 a1 120k'][i] < 0.97 for i in range(2)]
    m = [None if None in (pick(i, ('K4 a0.25 60k', 'K4 a1 60k')), pick(i, ('K1 a1 120k', 'K1 a4 120k'))) else
         pick(i, ('K4 a0.25 60k', 'K4 a1 60k')) - pick(i, ('K1 a1 120k', 'K1 a4 120k')) for i in range(2)]
    margin[arm] = m
    S[arm + '_margin'] = m
    V[f'L0 {arm}'] = verdict(l0)
    V[f'L-cap {arm}'] = verdict(cap)
    pre_ok = V[f'L0 {arm}'] == 'holds' and V[f'L-cap {arm}'] == 'holds'
    V[f'L1 {arm}'] = (verdict([None if x is None else x >= 0.01 for x in m]) if pre_ok else
                      'pending' if 'pending' in (V[f'L0 {arm}'], V[f'L-cap {arm}']) else 'not testable (precondition)')
  L += table(['arm', 'K1 a1 60k', 'K1 a4 60k', 'K1 a1 120k', 'K1 a4 120k', 'K4 a0.25 60k', 'K4 a1 60k', 'K1 a0 (off) 60k'], rows)
  L.append('\nMargin = best K = 4 at 60k − best K = 1 at 120k: ' +
           '; '.join(f'{a}: {" / ".join(pts(x) for x in m)}' for a, m in margin.items()))
  c, gg = margin['cos_w256_s4_N32k'], margin['geglu_w85_zinit_N32k']
  if V['L1 cos_w256_s4_N32k'] == 'holds':
    V['L2'] = verdict([None if None in (c[i], gg[i]) else c[i] - gg[i] >= 0.01 for i in range(2)])
  else:
    V['L2'] = 'refuted (L1 does not hold for Cos)' if V['L1 cos_w256_s4_N32k'] in ('refuted', 'split') else \
              V['L1 cos_w256_s4_N32k']
  return L, S, V


def main():
  p = argparse.ArgumentParser()
  for k in ('level', 'gate', 'why', 'grid', 'level2', 'width_old', 'infiloop_old', 'out'):
    p.add_argument(f'--{k}', default=None)
  args = p.parse_args()
  level, why, gate = load(args.level), load(args.why), load(args.gate)
  LA, SA, VA = section_a(level)
  LB, SB, VB = section_b(gate, why, load(args.infiloop_old))
  LC, SC, VC = section_c(why, load(args.width_old) if args.width_old else None)
  LH, SH, VH = section_h(load(args.why, args.grid))
  LL, SL, VL = section_l(load(args.level2))
  V = VA | VB | VC | VH | VL
  lines = LA + LB + LC + LH + LL + ['\n## Verdicts\n'] + table(['prediction', 'verdict'], [[k, v] for k, v in V.items()])
  print('\n'.join(lines))
  if args.out:
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, 'tables.md'), 'w').write('\n'.join(lines) + '\n')
    json.dump({'A': SA, 'B': SB, 'C': SC, 'H': SH, 'L': SL, 'verdicts': V}, open(os.path.join(args.out, 'summary.json'), 'w'),
              indent=1, default=str)


if __name__ == '__main__':
  main()
