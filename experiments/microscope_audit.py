"""Blind manual audit of last-layer neurons from the essay's own public microscopes.

The essay (Cammarata 2026, Sec. "better tools") puts the share of easily interpretable last-layer neurons at about 50%
for InceptionV1 mixed5b with modern tools and about 99% for the parameter-matched ConvNeXt-SwiGLU ('medium', 7,057,377
parameters, 90 epochs). Both microscopes are public: per neuron an example sheet (JPEG grid of crops centred on the
strongest position) and the dataset label, value and rank of every cell. This script samples neurons from both, cuts
each sheet into its cells and rebuilds them in one common layout, so that a rater cannot tell the model from the
format, and holds out some strong cells for a test of each description.

Cells kept (both microscopes have them): the top 16, 8 around the 99.9th percentile, 8 around the 99th percentile, and
the 4 most negative (the ConvNeXt sheet has 4; InceptionV1 has 16, of which the 4 most negative are used; InceptionV1's
negatives are pre-ReLU and do not reach the next layer, ConvNeXt's SwiGLU negatives do). The ConvNeXt 99.99% row is left
out (InceptionV1 has none). Shown to the rater (28 cells): top-16 ranks other than 3, 7, 11, 15; 99.9% cells 1, 3, 5, 7
(1-based: the even ones are held out); all 8 of the 99% cells; the 4 negatives. Held out (8 cells): top-16 ranks 3, 7,
11, 15 and 99.9% cells 2, 4, 6, 8. Test panel: the 8 held-out cells and 8 distractors (one strong cell each from 8 other
sampled neurons of the same model and layer), shuffled. A judge who sees only the rater's description picks 8 cells;
chance is 4 of 8 correct.

Relative activation of a cell = its value / the neuron's strongest value (the top-1 cell), so both scales read alike.

  python3 -m experiments.microscope_audit fetch --n 40 --seed 0 --out AUDIT_DIR
  python3 -m experiments.microscope_audit compose --out AUDIT_DIR
  python3 -m experiments.microscope_audit collect --out AUDIT_DIR --journal WORKFLOW_DIR/journal.jsonl --ratings ratings.json
  python3 -m experiments.microscope_audit score --out AUDIT_DIR --ratings ratings.json
Downloads go to AUDIT_DIR/raw (untrusted data: images and JSON are only parsed, never run).
"""

import argparse
import json
import os
import subprocess

import numpy as np

MICRO = {
  'medium': dict(
    base='https://already-clean.goodfire.com/thread/microscopes/leaky-relu-curve-basis/'
         'convnext-capacity-polysemanticity/medium/investigation/epoch_090/microscope-50k',
    layer='stage3_block2', sheet='examples/{layer}/{channel:04d}.jpg', data='data/{layer}.json', cell=96),
  'iv1': dict(
    base='https://already-clean.goodfire.com/microscope/inceptionv1',
    layer='mixed5b', sheet='sheets/{layer}/{channel}.jpg', data='data/{layer}.json', cell=112),
}
CELL = 128                                   # common cell size (px)
COLS = 8
HOLD_TOP = (2, 6, 10, 14)                    # 0-based ranks of the top 16 held out
HOLD_P999 = (1, 3, 5, 7)                     # 0-based 99.9% cells held out
SECTIONS = (('top', 'A. Strongest activations (12 of the top 16)'),
            ('p999', 'B. Around the 99.9th percentile (4 cells)'),
            ('p99', 'C. Around the 99th percentile: weak activations (8 cells)'),
            ('neg', 'D. Most negative activations (4 cells)'))


def curl(url, path):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  if not os.path.exists(path):
    subprocess.run(['curl', '-sSf', '-o', path, url], check=True)
  return path


def neuron_cells(model, n):
  """Cells of neuron record n as {bucket: [(label, value, (row, col))]}, in sheet order."""
  if model == 'medium':
    ex = {e['bucketId']: [(c['datasetLabel'], float(c['value'])) for c in e['cells']] for e in n['examples']}
    pos = {'top': [(0, i) for i in range(16)], 'p999': [(1, 8 + i) for i in range(8)],
           'p99': [(2, i) for i in range(8)], 'neg': [(2, 8 + i) for i in range(4)]}
    src = {'top': ex['top16'], 'p999': ex['p999'], 'p99': ex['p99'], 'neg': ex['bottom4']}
  else:
    b = n['b']                                                    # top16, p999, p99, neg16 as [index, label, value, rank]
    pos = {'top': [(0, i) for i in range(16)], 'p999': [(1, i) for i in range(8)],
           'p99': [(1, 8 + i) for i in range(8)], 'neg': [(2, i) for i in range(4)]}
    src = {k: [(c[1], float(c[2])) for c in b[j]] for j, k in enumerate(('top', 'p999', 'p99', 'neg'))}
    src['neg'] = src['neg'][:4]
  return {k: [(lab, val, rc) for (lab, val), rc in zip(src[k], pos[k])] for k in pos}


def fetch(a):
  rng = np.random.default_rng(a.seed)
  sample = {}
  for model, m in MICRO.items():
    data = json.load(open(curl(f"{m['base']}/{m['data'].format(layer=m['layer'])}",
                               os.path.join(a.out, 'raw', model, f"{m['layer']}.json"))))
    chans = sorted(int(c) for c in rng.choice(len(data['neurons']), a.n, replace=False))
    for c in chans:
      curl(f"{m['base']}/{m['sheet'].format(layer=m['layer'], channel=c)}",
           os.path.join(a.out, 'raw', model, 'sheets', f'{c:04d}.jpg'))
    sample[model] = chans
    print(model, len(chans), 'sheets', flush=True)
  json.dump({'seed': a.seed, 'n': a.n, 'sample': sample}, open(os.path.join(a.out, 'sample.json'), 'w'), indent=1)


def cut(sheet, cell, rc):
  r, c = rc
  return sheet.crop((c * cell, r * cell, (c + 1) * cell, (r + 1) * cell)).resize((CELL, CELL))


def numbered(img, k, font):
  from PIL import ImageDraw
  d = ImageDraw.Draw(img)
  d.rectangle((0, 0, 22, 14), fill=(0, 0, 0))
  d.text((2, 1), str(k), fill=(255, 255, 255), font=font)
  return img


def compose(a):
  from PIL import Image, ImageDraw, ImageFont
  font = ImageFont.load_default()
  S = json.load(open(os.path.join(a.out, 'sample.json')))
  rng = np.random.default_rng(S['seed'] + 1)
  items, key = [], {}
  strong = {}                                                     # model -> [(channel, cell image)] for distractors
  recs = {}
  for model, chans in S['sample'].items():
    m = MICRO[model]
    data = json.load(open(os.path.join(a.out, 'raw', model, f"{m['layer']}.json")))
    for c in chans:
      cells = neuron_cells(model, data['neurons'][c])
      sheet = Image.open(os.path.join(a.out, 'raw', model, 'sheets', f'{c:04d}.jpg')).convert('RGB')
      imgs = {k: [cut(sheet, m['cell'], rc) for _, _, rc in v] for k, v in cells.items()}
      recs[model, c] = (cells, imgs)
      strong.setdefault(model, []).extend((c, im) for im in imgs['top'] + imgs['p999'])
  ids = rng.permutation(len(recs))
  os.makedirs(os.path.join(a.out, 'items'), exist_ok=True)
  for (model, c), idn in zip(recs, ids):
    cells, imgs = recs[model, c]
    item = f'n{idn:03d}'
    top1 = cells['top'][0][1]
    shown = {'top': [i for i in range(16) if i not in HOLD_TOP], 'p999': [i for i in range(8) if i not in HOLD_P999],
             'p99': list(range(8)), 'neg': list(range(4))}
    rows, text, k = [], [], 0
    for sec, title in SECTIONS:
      hdr = Image.new('RGB', (COLS * CELL, 20), (235, 235, 235))
      ImageDraw.Draw(hdr).text((4, 4), title, fill=(0, 0, 0), font=font)
      rows.append(hdr)
      ims = []
      for i in shown[sec]:
        k += 1
        lab, val, _ = cells[sec][i]
        ims.append(numbered(imgs[sec][i].copy(), k, font))
        text.append(f'{k:2d} [{sec}] {lab} (relative activation {val / top1:+.2f})')
      for r0 in range(0, len(ims), COLS):
        row = Image.new('RGB', (COLS * CELL, CELL), (255, 255, 255))
        for j, im in enumerate(ims[r0:r0 + COLS]):
          row.paste(im, (j * CELL, 0))
        rows.append(row)
    panel = Image.new('RGB', (COLS * CELL, sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for r in rows:
      panel.paste(r, (0, y))
      y += r.height
    panel.save(os.path.join(a.out, 'items', f'{item}_panel.png'))
    held = [imgs['top'][i] for i in HOLD_TOP] + [imgs['p999'][i] for i in HOLD_P999]
    others = [x for x in strong[model] if x[0] != c]
    other_ch = rng.choice(sorted({ch for ch, _ in others}), 8, replace=False)
    distract = []
    for ch in other_ch:
      pool = [im for cc, im in others if cc == ch]
      distract.append(pool[rng.integers(len(pool))])
    order = rng.permutation(16)                                   # test cell t shows source order[t]
    tests = held + distract
    tp = Image.new('RGB', (COLS * CELL, 2 * CELL), (255, 255, 255))
    for t, s in enumerate(order):
      tp.paste(numbered(tests[s].copy(), t + 1, font), ((t % COLS) * CELL, (t // COLS) * CELL))
    tp.save(os.path.join(a.out, 'items', f'{item}_test.png'))
    neg_ratio = cells['neg'][0][1] / top1
    items.append({'item': item, 'panel': f'items/{item}_panel.png', 'test': f'items/{item}_test.png',
                  'cells': text, 'neg_over_pos': round(neg_ratio, 2)})
    key[item] = {'model': model, 'channel': c, 'held_out_test_cells': sorted(int(t) + 1 for t in np.flatnonzero(order < 8)),
                 'distractor_channels': [int(x) for x in other_ch]}
  items.sort(key=lambda r: r['item'])
  json.dump(items, open(os.path.join(a.out, 'items.json'), 'w'), indent=1)
  json.dump(key, open(os.path.join(a.out, 'key.json'), 'w'), indent=1)
  print(len(items), 'items', flush=True)


INTERP = ('one_job', 'one_job_related_offshoots')
NONARB = INTERP + ('two_related_jobs',)


def wilson(k, n, z=1.96):
  if n == 0:
    return float('nan'), float('nan')
  p = k / n
  den = 1 + z * z / n
  mid = (p + z * z / (2 * n)) / den
  half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return mid - half, mid + half


def score(a):
  key = json.load(open(os.path.join(a.out, 'key.json')))
  R = json.load(open(a.ratings))                                  # [{item, rater, pos_category, neg_category, picks}]
  out = {}
  for model in MICRO:
    rows = [r for r in R if key[r['item']]['model'] == model]
    res = {'ratings': len(rows)}
    for name, ok in (('interpretable', INTERP), ('non_arbitrary', NONARB)):
      k = sum(r['pos_category'] in ok for r in rows)
      res[f'pos_{name}'] = k / max(len(rows), 1)
      res[f'pos_{name}_ci95'] = wilson(k, len(rows))
    res['pos_categories'] = {c: sum(r['pos_category'] == c for r in rows) for c in sorted({r['pos_category'] for r in rows})}
    res['neg_categories'] = {c: sum(r.get('neg_category') == c for r in rows)
                             for c in sorted({r.get('neg_category') for r in rows if r.get('neg_category')})}
    hits = [len(set(r['picks']) & set(key[r['item']]['held_out_test_cells'])) for r in rows if r.get('picks')]
    res['test_hits_mean'] = float(np.mean(hits)) if hits else float('nan')   # of 8; chance 4
    res['test_hits_by_category'] = {c: float(np.mean([len(set(r['picks']) & set(key[r['item']]['held_out_test_cells']))
                                                      for r in rows if r['pos_category'] == c and r.get('picks')] or [np.nan]))
                                    for c in res['pos_categories']}
    by_item = {}
    for r in rows:
      by_item.setdefault(r['item'], []).append(r['pos_category'] in INTERP)
    pairs = [v[:2] for v in by_item.values() if len(v) >= 2]
    if pairs:
      x = np.array(pairs, float)
      po = float((x[:, 0] == x[:, 1]).mean())
      p1, p2 = x[:, 0].mean(), x[:, 1].mean()
      pe = p1 * p2 + (1 - p1) * (1 - p2)
      res['rater_agreement'] = po
      res['cohen_kappa'] = (po - pe) / (1 - pe) if pe < 1 else float('nan')
    out[model] = res
  print(json.dumps(out, indent=1))
  if a.save:
    json.dump(out, open(a.save, 'w'), indent=1)


def collect(a):
  """Ratings from a workflow journal: agents labelled rate:{item}:{rater} (rating fields) and test:{item}:{rater} (picks)."""
  rate, test = {}, {}
  for line in open(a.journal):
    e = json.loads(line)
    if e.get('type') != 'result' or not isinstance(e.get('result'), dict):
      continue
    kind, item, rater = (e.get('label') or '::').split(':')[:3]
    (rate if kind == 'rate' else test if kind == 'test' else {})[item, rater] = e['result']
  rows = []
  for (item, rater), r in sorted(rate.items()):
    t = test.get((item, rater), {})
    picks = t.get('picks')
    if picks is not None and (len(set(picks)) != 8 or not all(1 <= x <= 16 for x in picks)):
      picks = None                                                # malformed answer: no test score
    rows.append({**r, 'item': item, 'rater': rater, 'picks': picks, 'test_confidence': t.get('confidence')})
  json.dump(rows, open(a.ratings, 'w'), indent=1)
  print(len(rows), 'ratings,', sum(r['picks'] is not None for r in rows), 'with a valid test', flush=True)


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('cmd', choices=['fetch', 'compose', 'collect', 'score'])
  p.add_argument('--out', required=True, help='audit directory')
  p.add_argument('--n', type=int, default=40, help='neurons per microscope')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--ratings', default=None, help='collect: output; score: input (JSON list of ratings)')
  p.add_argument('--journal', default=None, help='collect: the workflow journal.jsonl')
  p.add_argument('--save', default=None, help='score: write the summary here')
  a = p.parse_args()
  {'fetch': fetch, 'compose': compose, 'collect': collect, 'score': score}[a.cmd](a)


if __name__ == '__main__':
  main()
