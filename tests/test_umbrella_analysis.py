"""E6 umbrella analysis on a CPU toy (experiments.convnext_umbrella_analysis, experiments.convnext_umbrella_arms).

The toy: ConvNeXt-SwiGLU 'tiny' at 32 px with 8 classes in float64. Pretraining with per-class attribution of gate and
value (E1's role), a torch AdamW warm-up (the branch), then fine-tuning runs written in the file layout of
experiments/convnext_umbrella.py: groups 0-7 the classes, 8 the photo, 9 hist; 2 shadow splits of the 15 replay slots
into 8 slot groups (the same in taught and untaught runs); frozen-v or standard AdamW; sums at every phase-1 step and
every 2 steps, shadows and full states at the same points, tau fixed at 4 and phase 2 of 6 steps.
"""

import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import convnext_umbrella_analysis as UA
from experiments import convnext_umbrella_arms as UR
from experiments.convnext_train import param_groups
from rcdl.attribution import GroupAttribution, snapshot_from_torch_adamw
from rcdl.convnext import build_convnext

K, HW, NREP, TAU, PH2, EVERY, NSH = 8, 32, 15, 4, 6, 2, 2
T, S, A_, B_ = 1, 2, 3, 4
RELB, RELU, OTH = [5], [6], [0, 7]
PHOTO, HIST = K, K + 1
DT = torch.float64
LR = 3e-3


@pytest.fixture(scope='module', autouse=True)
def one_thread():
  """One intra-op thread in this module (small tensors; no oversubscription when other jobs share the CPUs)."""
  n = torch.get_num_threads()
  torch.set_num_threads(1)
  yield
  torch.set_num_threads(n)


def toy_model(seed=0):
  torch.manual_seed(seed)
  m = build_convnext('tiny', n_classes=K).to(DT)
  with torch.no_grad():
    for mod in m.modules():
      if isinstance(mod, nn.LayerNorm):
        mod.weight.copy_(1 + 0.1 * torch.randn_like(mod.weight))
        mod.bias.copy_(0.05 * torch.randn_like(mod.bias))
      elif isinstance(mod, (nn.Linear, nn.Conv2d)):
        mod.weight.normal_(0, 0.2)
        mod.bias.normal_(0, 0.05)
    for n, b in m.blocks():
      b.gamma.copy_(0.3 + 0.05 * torch.randn_like(b.gamma))
  return m


def replay(seed, step):
  g = torch.Generator().manual_seed(1000 * seed + step + 7)
  x = torch.randn(NREP, 3, HW, HW, generator=g, dtype=DT)
  rest = torch.tensor([0, 5, 6, 7])[torch.randint(0, 4, (7,), generator=g)]
  return x, torch.cat([torch.tensor([T, T, S, S, A_, A_, B_, B_]), rest])


def slot_groups(seed, step, r):
  perm = np.random.default_rng([seed, 2_000_000 + 8 * step + r]).permutation(NREP)
  g = np.empty(NREP, np.int64)
  g[perm] = 1 + np.arange(NREP) * 8 // NREP
  return torch.from_numpy(g)


def photo_x():
  return torch.randn(1, 3, HW, HW, generator=torch.Generator().manual_seed(99), dtype=DT)


def pretrain(steps=6, lr=1e-2):
  """E1's role: gate and value attributed per class, the rest trained by torch AdamW."""
  m = toy_model()
  maps = {f'{n}.{k}': getattr(b, k) for n, b in m.blocks() for k in ('pw_gate', 'pw_value')}
  attr = GroupAttribution(maps, n_groups=K, optimizer='adamw', lr=lr, weight_decay=0.05, dtype=DT)
  opt = torch.optim.AdamW(param_groups(m, [p for l in maps.values() for p in (l.weight, l.bias)], 0.05), lr=lr)
  g = torch.Generator().manual_seed(1)
  for _ in range(steps):
    x, y = torch.randn(16, 3, HW, HW, generator=g, dtype=DT), torch.arange(16) % K
    opt.zero_grad()
    F.cross_entropy(m(x), y).backward()
    attr.step(y)
    opt.step()
  state = {k: v.detach().clone() for k, v in m.state_dict().items()}
  comp = attr.compact_state()
  init = {k: {'W0': tr.W0.clone(), 'b0': tr.b0.clone()} for k, tr in attr.tracked.items()}
  probe = torch.randn(2 * K, 3, HW, HW, generator=g, dtype=DT)
  m2 = toy_model()
  m2.load_state_dict(state)
  _, st = UA.block_states(m2, probe)
  y = torch.arange(2 * K) % K
  atlas = {b: {'R': torch.stack([v['u'][y == c].max(1).values.mean(0) for c in range(K)], 1)} for b, v in st.items()}
  return {'state': state, 'attr': comp, 'init': init, 'atlas': atlas}


def warmup(state, steps=3, lr=5e-3):
  m = toy_model()
  m.load_state_dict(state)
  opt = torch.optim.AdamW([{'params': [p for p in m.parameters() if p.dim() >= 2], 'weight_decay': 0.05},
                           {'params': [p for p in m.parameters() if p.dim() < 2], 'weight_decay': 0.0}], lr=lr)
  for s in range(steps):
    for gr in opt.param_groups:
      gr['lr'] = lr * (s + 1) / steps
    x, y = replay(10_000, s)
    opt.zero_grad()
    (0.5 * F.cross_entropy(m(x), y)).backward()
    opt.step()
  pname = {id(p): n for n, p in m.named_parameters()}
  return {'model': {k: v.detach().clone() for k, v in m.state_dict().items()},
          'snap': snapshot_from_torch_adamw(opt, UA.tracked_maps(m)),
          'states': {pname[id(p)]: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in opt.state[p].items()}
                     for p in m.parameters()}}


def finetune(out, br, mode, opt_kind='frozen_v', seed=0, full_saves=(), sel_path='', untaught_steps=TAU + PH2):
  """The loop and saves of experiments/convnext_umbrella.py cmd_run, on the toy."""
  os.makedirs(out, exist_ok=True)
  m = toy_model()
  m.load_state_dict(br['model'])
  m.train()
  maps = UA.tracked_maps(m)
  hp = dict(optimizer='adamw', lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, bias_weight_decay=0.0, dtype=DT)
  attr = GroupAttribution(maps, n_groups=K + 2, **hp)
  shadows = [GroupAttribution(maps, n_groups=10, shadow=True, **hp) for _ in range(NSH)]
  attr.load_snapshot(br['snap'], history_group=HIST)
  for sh in shadows:
    sh.load_snapshot(br['snap'], history_group=9)
  tracked = {id(p) for l in maps.values() for p in (l.weight, l.bias)}
  untracked = [(n, p) for n, p in m.named_parameters() if id(p) not in tracked]
  opt_unt = None
  if opt_kind == 'frozen_v':
    attr.freeze_second_moment()
    for sh in shadows:
      sh.freeze_second_moment()
    for _, p in untracked:
      p.requires_grad_(False)
  else:
    opt_unt = torch.optim.AdamW([{'params': [p for _, p in untracked if p.dim() >= 2], 'weight_decay': 0.05},
                                 {'params': [p for _, p in untracked if p.dim() < 2], 'weight_decay': 0.0}], lr=LR)
    for n, p in untracked:
      opt_unt.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in br['states'][n].items()}
  named = {'P': [PHOTO], 'hist': [HIST], 'BEE': [T], 'UMB': [S], 'A': [A_], 'B': [B_], 'RELb': RELB, 'RELu': RELU,
           'OTH': OTH}
  cfg = {'mode': mode, 'opt': opt_kind, 'seed': seed, 'lr': LR, 'wd': 0.05, 'arm': 'toy', 'photo': 'toy:0',
         'phase2': PH2, 'target': T, 'replay': 'toy', 'rest': 7, 'rest_exclude': '', 'sel': sel_path, 'sets': named}
  json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'))
  full_saves = set(full_saves)

  def save(s, full=False, shadow=False, tag=None):
    torch.save(m.state_dict(), os.path.join(out, f'model_{s:04d}.pt'))
    torch.save(attr.group_sums(named), os.path.join(out, f'sums_{s:04d}.pt'))
    if shadow:
      torch.save([sh.compact_state() for sh in shadows], os.path.join(out, f'shadow_{s:04d}.pt'))
    if full:
      torch.save(attr.compact_state(), os.path.join(out, f'attr_{s:04d}.pt'))
    if tag:
      torch.save({'step': s, 'model': m.state_dict(), 'attr': attr.snapshot()}, os.path.join(out, f'snap_{tag}.pt'))

  torch.save({k: {'W0': tr.W0, 'b0': tr.b0} for k, tr in attr.tracked.items()}, os.path.join(out, 'attr_init.pt'))
  save(0, full=True, shadow=True)
  photo = photo_x() if mode != 'untaught' else None
  phase, tau, s = (1 if mode != 'untaught' else 2), None, 0
  log = open(os.path.join(out, 'log.jsonl'), 'w')
  while True:
    if mode == 'untaught' and s >= untaught_steps:
      break
    if mode != 'untaught' and phase == 2 and (mode == 'solo' or s >= tau + PH2):
      break
    xs, ys, gs = [], [], []
    if phase == 1:
      xs.append(photo), ys.append(torch.tensor([T])), gs.append(torch.tensor([PHOTO]))
    if mode != 'solo':
      xr, yr = replay(seed, s)
      xs.append(xr), ys.append(yr), gs.append(yr)
    x, y, g = torch.cat(xs), torch.cat(ys), torch.cat(gs)
    n_rep = len(y) - (1 if phase == 1 else 0)
    w = torch.full((len(y),), 0.5 / max(n_rep, 1), dtype=DT)
    if phase == 1:
      w[0] = 0.5 if mode != 'solo' else 1.0
    loss = (w * F.cross_entropy(m(x), y, reduction='none')).sum()
    if opt_unt is not None:
      opt_unt.zero_grad(set_to_none=True)
    loss.backward()
    attr.step(g, LR)
    for r, sh in enumerate(shadows):
      sg = slot_groups(seed, s, r) if mode != 'solo' else torch.zeros(0, dtype=torch.long)
      if phase == 1:
        sg = torch.cat([torch.zeros(1, dtype=torch.long), sg])
      sh.step(sg, LR)
    if opt_unt is not None:
      opt_unt.step()
    s += 1
    if photo is not None:
      with torch.no_grad():
        log.write(json.dumps({'step': s, 'phase': phase, 'top1': int(m(photo)[0].argmax())}) + '\n')
    if phase == 1 and s >= TAU:
      tau = s
      json.dump({'tau': tau, 'censored': False}, open(os.path.join(out, 'tau.json'), 'w'))
      save(s, full=True, shadow=True, tag='tau')
      torch.save({k: (tr.C[PHOTO].clone(), tr.Cb[PHOTO].clone()) for k, tr in attr.tracked.items()},
                 os.path.join(out, 'photo_parts_tau.pt'))
      phase = 2
      continue
    if phase == 1 or s % EVERY == 0 or s in full_saves or (tau is not None and s == tau + PH2):
      full = s in full_saves or (tau is not None and s == tau + PH2)
      save(s, full=full, shadow=(s % EVERY == 0) or full)
  save(s, full=True, shadow=True)
  log.close()
  for at in [attr] + shadows:
    for tr in at.tracked.values():
      tr.handle.remove()
  return out


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
  root = str(tmp_path_factory.mktemp('umb'))
  e1 = pretrain()
  br = warmup(e1['state'])
  runs = {'taught': finetune(os.path.join(root, 'taught'), br, 'taught'),
          'untaught': finetune(os.path.join(root, 'untaught'), br, 'untaught', full_saves=(TAU, TAU + PH2)),
          'solo': finetune(os.path.join(root, 'solo'), br, 'solo'),
          'std_taught': finetune(os.path.join(root, 'std_taught'), br, 'taught', opt_kind='standard')}
  return {'root': root, 'e1': e1, 'br': br, 'runs': runs}


def keys_of():
  m = toy_model()
  maps = UA.tracked_maps(m)
  return m, maps, UA.map_keys(m, maps)


def loader_for(run, model, keys, **kw):
  return UA.make_loader(run, model, keys, DT, 'cpu', **kw)


# ------------------------------------------------------------------------------------------------ buckets

def test_bucket_completeness_over_a_piecewise_path(toy):
  run = toy['runs']['taught']
  _, maps, keys = keys_of()
  verts = UA.path_vertices(run, 0, TAU + PH2, shadows=True)
  assert verts[0] == 0 and verts[-1] == TAU + PH2 and TAU in verts
  th = {s: torch.load(os.path.join(run, f'model_{s:04d}.pt')) for s in verts}
  whole = UA.load_buckets(run, 0, TAU + PH2, keys, shadows=True)
  names = whole['names']
  acc = {k: torch.zeros_like(whole['W'][k][:len(names)]) for k in maps}
  for s1, s2 in UA.segments_of(verts):
    bk = UA.load_buckets(run, s1, s2, keys, shadows=True)
    assert bk['names'] == names and len(bk['extra']) == 8 * NSH
    assert bk['check']['rem_vs_sums'] < 1e-6 and bk['check']['shadow_sum'] < 1e-6
    assert bk['check']['shadow_vs_replay'] < 5e-3                       # fp16 storage of the shadow parts
    for k, pre in keys.items():
      dW = th[s2][pre + '.weight'] - th[s1][pre + '.weight']
      tot = bk['W'][k][:len(names)].sum(0)
      assert torch.allclose(tot, dW, atol=1e-15, rtol=0)                # exact in float64
      assert bk['W'][k][names.index('R')].abs().max() <= 1e-5 * dW.abs().max() + 1e-12   # the remainder is rounding
      assert torch.allclose(bk['b'][k][:len(names)].sum(0), th[s2][pre + '.bias'] - th[s1][pre + '.bias'], atol=1e-15)
      acc[k] += bk['W'][k][:len(names)]
  # over the path, a set's writes plus its decay telescope to the whole interval's
  iP, iD = names.index('P'), names.index('dec_P')
  rest = [names.index(n) for n in names if n not in ('P', 'dec_P', 'dec_0', 'R')]
  for k in maps:
    sc = whole['W'][k].abs().max()
    assert torch.allclose(acc[k][iP] + acc[k][iD], whole['W'][k][iP] + whole['W'][k][iD], atol=1e-6 * sc)
    assert torch.allclose(acc[k][rest].sum(0), whole['W'][k][rest].sum(0), atol=1e-6 * sc)
    assert acc[k][iP].abs().max() > 0
  assert not UA.load_buckets(run, 0, TAU, keys)['unt']                   # frozen-v: untracked parameters do not move


# ---------------------------------------------------------------------------------------- integrated gradients

def test_capture_gives_per_example_gradients():
  m, maps, _ = keys_of()
  x = torch.randn(3, 3, HW, HW, generator=torch.Generator().manual_seed(5), dtype=DT)
  fn = lambda z: z[:, T] - z[:, S]
  for p in m.parameters():
    p.requires_grad_(False)
  cap = UA.GradCapture(maps)
  _, grads = UA._forward_grads(m, cap, x, {'m': fn})
  cap.remove()
  for name in ('stage0.block0.pw_gate', 'stage2.block4.pw_down', 'head'):
    lin = maps[name]
    lin.weight.requires_grad_(True)
    lin.bias.requires_grad_(True)
    for j in range(3):
      gw, gb = torch.autograd.grad(fn(m(x[j:j + 1])).sum(), [lin.weight, lin.bias])
      assert torch.allclose(grads['m'][name][0][j], gw, atol=1e-13)
      assert torch.allclose(grads['m'][name][1][j], gb, atol=1e-13)
    lin.weight.requires_grad_(False)
    lin.bias.requires_grad_(False)


def test_ig_completeness_and_rows(toy):
  run = toy['runs']['taught']
  m, maps, keys = keys_of()
  x = torch.cat([photo_x(), torch.randn(3, 3, HW, HW, generator=torch.Generator().manual_seed(3), dtype=DT)])
  fns = UA._fns(T, S)
  verts = UA.path_vertices(run, 0, TAU + PH2, shadows=False)
  res = UA.param_ig(m, maps, UA.segments_of(verts), loader_for(run, m, keys), x, fns, nodes=16, rows=('P', 'BEE'),
                    tol=1e-9, check=range(4))
  for k in fns:
    assert res[k]['eps'].max() <= 1e-8, (k, res[k]['eps'])
  # the margin's IG is zt's minus zs's, bucket by bucket
  for mp in maps:
    assert np.allclose(res['m']['ig'][mp], res['zt']['ig'][mp] - res['zs']['ig'][mp], atol=1e-12)
  names = res['names']
  rsel = [names.index('P'), names.index('BEE')]
  for mp, rows in res['m']['rows'].items():
    tot = rows.sum(-1) + (res['m']['down_bias'][mp] if mp.endswith('pw_down') else 0.0)
    assert np.allclose(tot, res['m']['ig'][mp][:, rsel], atol=1e-13), mp
    if mp.endswith('pw_down'):
      assert rows.shape[-1] == maps[mp].in_features
    else:
      assert rows.shape[-1] == maps[mp].out_features
  assert abs(UA.ig_images(res['m'], names, 'P')[0]) > 0


def test_ig_completeness_with_untracked_parameters(toy):
  """Standard optimizer: the untracked parameters move; their bucket is integrated by forward-mode JVP."""
  run = toy['runs']['std_taught']
  m, maps, keys = keys_of()
  x = torch.cat([photo_x(), torch.randn(2, 3, HW, HW, generator=torch.Generator().manual_seed(4), dtype=DT)])
  verts = UA.path_vertices(run, 0, TAU + PH2, shadows=False)
  res = UA.param_ig(m, maps, UA.segments_of(verts), loader_for(run, m, keys), x, {'m': UA._margin(T, S)}, nodes=16,
                    tol=1e-9, check=range(3))
  assert res['names'][-1] == 'unt' and np.abs(res['m']['unt']).max() > 1e-6
  assert res['m']['eps'].max() <= 1e-8


# -------------------------------------------------------------------------------------------- pseudo-groups

def test_did_z_on_synthetic_numbers():
  r = UA.did_z(6.0, [1.0, 2.0, 3.0, 4.0, 5.0])
  assert r['mean_pseudo'] == 3.0 and abs(r['sd_pseudo'] - np.sqrt(2.5)) < 1e-12
  assert abs(r['z'] - 3.0 / np.sqrt(2.5)) < 1e-12 and r['n_pseudo'] == 5
  rng = np.random.default_rng(0)
  pseudo = 0.4 + 0.1 * rng.standard_normal(64)                         # a random group's DiD is not 0 on average
  assert abs(UA.did_z(float(pseudo.mean()), pseudo)['z']) < 1e-9       # z is centred on the pseudo mean
  assert UA.did_z(float(pseudo.mean() + 5 * pseudo.std(ddof=1)), pseudo)['z'] == pytest.approx(5.0)
  assert np.isnan(UA.did_z(1.0, [2.0])['z'])


@pytest.fixture(scope='module')
def toy_analysis(toy):
  vsets = {}
  g = torch.Generator().manual_seed(11)
  for name, n in (('V_ref', 12), ('V_umb', 6), ('V_bee', 4)):
    vsets[name] = {'x': torch.randn(n, 3, HW, HW, generator=g, dtype=DT), 'idx': list(range(n)), 'y': [0] * n}
  ctx = {'make_model': lambda: build_convnext('tiny', n_classes=K), 'device': 'cpu', 'dtype': DT, 'vdtype': DT,
         'x_P': photo_x(), 'target': T, 'source': S, 'kind': 'photo', 'rel_target': RELB, 'vsets': vsets,
         'e1': {k: toy['e1'][k] for k in ('atlas', 'attr', 'init', 'state')}, 'prov': None, 'nodes': 8, 'vnodes': 8,
         'step_nodes': 4, 'tol': 1e-6, 'max_nodes': 64, 'batch': 16, 'vbatch': 8, 'perm': 200, 'n_near': 5,
         'n_mp': 3, 'q_sel': 0.9, 'rows': ('P', 'BEE'), 'poles': 2, 'per_class': True}
  summ, arrays = UA.analyze(toy['runs']['taught'], ctx, toy['runs']['untaught'])
  solo, _ = UA.analyze(toy['runs']['solo'], ctx)
  return summ, arrays, solo


def test_pseudo_group_did_machinery_on_the_toy(toy_analysis):
  summ, arrays, _ = toy_analysis
  M1 = summ['M1']
  assert M1['paths']['I1'] == [0, 2, 4] and M1['paths']['I2'] == [4, 6, 8, 10]
  for I in ('I1', 'I2'):
    assert M1['completeness'][f'taught_{I}']['eps_P'] <= 1e-6
    assert M1['completeness'][f'untaught_{I}']['eps_P'] <= 1e-6
    pseudo = arrays['M1_pseudo_DiD'][I]['total']
    assert pseudo.shape == (8 * NSH,)
    replay = sum(M1['DiD'][I][n]['total'] for n in ('BEE', 'UMB', 'A', 'B', 'RELb', 'RELu', 'OTH'))
    for r in range(NSH):                                  # each shadow's 8 slot groups split the same replay writes
      assert abs(pseudo[8 * r:8 * r + 8].sum() - replay) <= 1e-2 * abs(replay) + 1e-9
    z = M1['z'][I]['BEE']['blocks']
    blocks = arrays['M1_pseudo_DiD'][I]['blocks']
    assert z['z'] == pytest.approx((M1['DiD'][I]['BEE']['blocks'] - blocks.mean()) / blocks.std(ddof=1))
  assert M1['DiD']['I1']['P']['total'] == pytest.approx(M1['F']['taught']['I1']['m']['P']['total'])  # untaught: no photo
  assert M1['F']['untaught']['I1']['m']['P']['total'] == 0.0
  dm = summ['m_P']
  for who, I, a, b in (('taught', 'I1', '0', 'tau'), ('taught', 'I2', 'tau', 'end')):
    tot = sum(v['total'] for n, v in M1['F'][who][I]['m'].items())
    assert tot == pytest.approx(dm[b] - dm[a], rel=1e-5)
  assert np.isfinite(M1['shelter']['E_OTH']) and np.isfinite(M1['feed']['phi_BEE_I2'])


def test_analysis_end_to_end_on_the_toy(toy_analysis):
  summ, arrays, solo = toy_analysis
  ck = summ['checks']
  assert ck['fine_I1']['eps_P'] <= 1e-6 and ck['rows_vs_map'] <= 1e-12
  assert ck['own_upstream_resid'] <= 1e-10                             # the own-row / upstream split is exact
  assert ck['two_level_resid'] <= 1e-6                                 # E1's parts + init + R + warm-up = W_w
  assert ck['swap_all_identity'] <= 1e-12
  M4 = summ['M4']
  assert sum(M4['S'].values()) == pytest.approx(1.0, abs=1e-9)
  assert M4['swaps']['stage3']['tot']['m'] != M4['readout_w']['m']
  M2 = summ['M2']
  assert len(M2['G_t']) == TAU and np.isfinite(M2['L']) and np.isfinite(M2['K'])
  assert sum(M2['K_split'].values()) == pytest.approx(M2['N']['V_near'], rel=1e-4)
  assert ck['M2_eps_P']['I1'] <= 1e-6
  M3 = summ['M3']
  assert M3['n_C50'] >= 1 and 0 <= M3['own_share']['median_C50'] <= 1
  assert set(M3['q']) == {'B_pole', 'U_pole_fading'}
  q = arrays['q_t']
  assert q.min() >= 0 and q.max() <= 1
  assert len(summ['per_class']['I1']['top_feed']) == min(10, K)
  assert 'M1' not in solo and 'M3' not in solo and len(solo['M2']['G_t']) == TAU


def test_swaps_reproduce_the_photo_parts(toy):
  run = toy['runs']['taught']
  m, maps, keys = keys_of()
  live = torch.load(os.path.join(run, 'photo_parts_tau.pt'))
  bk = UA.load_buckets(run, 0, TAU, keys)
  ip = bk['names'].index('P')
  photo = {k: (bk['W'][k][ip], bk['b'][k][ip]) for k in maps}
  for k in maps:
    sc = live[k][0].abs().max()
    assert torch.allclose(photo[k][0], live[k][0], atol=1e-6 * sc) and sc > 0     # fp32 group sums
    assert torch.allclose(photo[k][1], live[k][1], atol=1e-6 * live[k][1].abs().max() + 1e-12)
  thw = torch.load(os.path.join(run, 'model_0000.pt'))
  tht = torch.load(os.path.join(run, f'model_{TAU:04d}.pt'))
  allm = list(maps)
  sw = UA.swap_states(thw, tht, photo, keys, allm)
  for k in thw:
    assert torch.allclose(sw['P'][k] + sw['-P'][k], thw[k] + tht[k], atol=1e-14)
    assert torch.equal(sw['tot'][k], tht[k])                      # frozen untracked: theta^tot(all) = theta_tau
  for k, pre in keys.items():
    assert torch.allclose(sw['P'][pre + '.weight'], thw[pre + '.weight'] + live[k][0], atol=1e-6 * live[k][0].abs().max())
  st3 = UA.swap_states(thw, tht, photo, keys, UA.swap_set_maps('stage3', allm))
  assert torch.equal(st3['P']['stages.2.0.pw_gate.weight'], thw['stages.2.0.pw_gate.weight'])
  assert not torch.equal(st3['P']['stages.3.0.pw_gate.weight'], thw['stages.3.0.pw_gate.weight'])
  for name in UA.SWAP_SETS:
    assert UA.swap_set_maps(name, allm)


# ------------------------------------------------------------------------------------------------ statistics

def test_statistics():
  rng = np.random.default_rng(0)
  assert UA.signed_rank_test(rng.normal(1.0, 0.3, 20), 2000)['p'] < 0.01
  assert UA.signed_rank_test(rng.normal(0.0, 1.0, 20), 2000)['p'] > 0.01
  assert UA.perm_mean_diff(rng.normal(1, 0.1, 10), rng.normal(0, 0.1, 30), 2000)['p'] < 0.01
  x = rng.normal(size=50)
  r = UA.slope_perm(x, 2 * x + 0.01 * rng.normal(size=50), 2000)
  assert abs(r['beta'] - 2) < 0.01 and r['p'] < 0.01
  assert list(UA.c50(np.array([5.0, -3.0, 1.0, 4.0, 0.5]))) == [0, 3]
  T_ = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
  assert np.allclose(UA.class_percentile(T_, np.array([1.0, -1.0]), 2), [1.0, 1.0])


def test_provenance_pi():
  units, C = 3, 5
  W = torch.randn(units, 2, C, generator=torch.Generator().manual_seed(0))
  Vm = torch.randn(units, C, generator=torch.Generator().manual_seed(1))
  res = {'b': {'value_writers': W, 'gate_writers': W, 'Vm': Vm, 'Gm': Vm, 'resid': 0.0}}
  out = UA.provenance_pi(res, 1, 3)
  for n in range(units):
    s1 = np.sign(Vm[n, 1].item())
    want = sum(s1 * W[n, 0, 1] > s1 * W[n, 0, k] for k in range(C) if k != 1) / (C - 1)
    assert out['blocks']['b']['pi_value_t'][n] == pytest.approx(float(want))


def test_eval_sets_exclude_candidates():
  labels = np.repeat(np.arange(6), 12)[np.random.default_rng(0).permutation(72)]
  arm = {'target': 1, 'source': 2, 'rel_target': [3], 'rel_source': [], 'excluded_from_eval': [],
         'photos': [{'photo': f'val:{int(np.flatnonzero(labels == 2)[9])}'}], 'reserve': [{'photo': 'train:5'}]}
  sel = {'arms': {'x': arm, 'y': {'excluded_from_eval': [int(np.flatnonzero(labels == 0)[1])], 'photos': []}}}
  ex = UA.candidate_exclusions([sel])
  v = UA.eval_set_indices(labels, arm, ex, first_n=10, n_classes=6)
  assert set(labels[v['V_ref']]) == {0, 4, 5} and len(v['V_ref']) == 29          # class 0 loses one candidate
  assert int(np.flatnonzero(labels == 2)[9]) not in v['V_umb'] and len(v['V_umb']) == 12 - 8 - 1
  assert len(v['V_bee']) == 12 - 8 and not set(v['V_ref']) & ex


# ------------------------------------------------------------------------------------------------ decisions

def _rec(seed, photo, z, did=1.0, dA=0.1, dB=0.1, e_oth=0.1, L=0.1, p=0.001, kind='photo', tag='B'):
  return {'tag': tag, 'kind': kind, 'seed': seed, 'photo': photo, 'target': T, 'arm': 'toy', 'opt': 'frozen_v',
          'M1': {'feed': {'DiD_BEE_I2': did, 'z_BEE_I2': z, 'DiD_A_I2': dA, 'DiD_B_I2': dB},
                 'shelter': {'E_OTH': e_oth}, 'umbrella': {I: {'blocks': {'z': z, 'did': did}} for I in ('I1', 'I2')},
                 'DiD': {'I2': {'UMB': {'blocks': did}}}},
          'M2': {'L': L, 'p_umb_ref': p}, 'M3': {}, 'M4': {}}


def test_decision_rules_on_synthetic_records():
  photos = [f'val:{i}' for i in range(5)]
  B = [_rec(sd, ph, 4.0) for sd in range(3) for ph in photos]
  F_ = [_rec(sd, c, 0.5, e_oth=0.5, L=0.7, kind='control', tag='F') for sd in range(3) for c in ('noise:0', 'shuffle:1')]
  F_ += [_rec(0, c, 0.5, e_oth=0.5, L=0.6, kind='control', tag='F') for c in ('noise:1', 'shuffle:2')]
  Bp = [_rec(0, ph, 3.5, tag='Bp') for ph in photos]
  lgo = [{'photo': ph, 'seed': sd, 'variant': v, 'm_P_end': {'base': 1.0, 'target': 1.5, 'source': 0.5, 'A': 1.0}[v]}
         for sd in (0, 1) for ph in photos for v in ('base', 'target', 'source', 'A')]
  out = UA.decide(B + F_ + Bp, lgo)
  assert out['P-C1a']['verdict'] == 'O' and out['P-C1b']['verdict'] == 'O' and out['C1'] == 'supported'
  assert all(out['P-C1a']['lgo_agreement']) and len(out['P-C1a']['lgo_agreement']) == 10
  assert out['P-C2']['calibrated'] and out['P-C2']['verdict'] == 'O'
  null = [_rec(sd, ph, 0.5, e_oth=0.9, L=0.65, p=0.5) for sd in range(3) for ph in photos]
  out = UA.decide(null + F_ + Bp, [])
  assert out['P-C1a']['verdict'] == 'against' and out['P-C1b']['verdict'] == 'against' and out['C1'] == 'contradicted'
  assert out['P-C2']['verdict'] == 'H'
  F_uncal = [dict(c, M2={'L': 0.1, 'p_umb_ref': 0.5}) for c in F_]
  assert UA.decide(B + F_uncal, [])['P-C2']['verdict'] == 'not testable in tiny'
  rode = [_rec(0, 'noise:0', 4.0, e_oth=0.0, kind='control', tag='F')]               # rode the bee route: not counted
  r = UA.rule_c1b([_rec(0, 'val:0', 4.0, e_oth=0.1)], rode + [_rec(0, 'shuffle:1', 0.1, e_oth=0.5, kind='control', tag='F')])
  assert r['per_run'][0]['O'] and r['per_run'][0]['rode_bee_route'] == ['noise:0']


# ------------------------------------------------------------------------------------------------ arm helpers

def test_lr_rule():
  rows = [{'lr': 1e-4, 'tau': 120, 'censored': False, 'drift': 0.1},
          {'lr': 3e-4, 'tau': 60, 'censored': False, 'drift': 0.4},
          {'lr': 1e-3, 'tau': 20, 'censored': False, 'drift': 0.8}, {'lr': 3e-3, 'tau': 8, 'censored': False, 'drift': 2.0}]
  r = UR.lr_rule(rows)
  assert r['lr_star'] == 1e-3 and not r['flagged']
  rows[2]['drift'] = 1.5
  assert UR.lr_rule(rows)['lr_star'] == 3e-4                    # tau <= 100 within the cap
  rows[1]['drift'] = 1.5
  r = UR.lr_rule(rows)
  assert r['lr_star'] == 3e-4 and r['flagged']                  # tau <= 100, cap missed
  assert UR.lr_rule([{'lr': 1e-4, 'tau': 300, 'censored': True, 'drift': 0.0}])['lr_star'] is None


def test_draw_triples_is_deterministic_and_follows_the_rule():
  rng = np.random.default_rng(1)
  n = 200
  acc = rng.uniform(0, 0.8, n)
  wn = rng.integers(6, 20, (n, n))
  wn = np.minimum(wn, wn.T)
  np.fill_diagonal(wn, 0)
  rank = rng.integers(1, 1000, (n, n))
  excl = {0, 1, 2}
  a = UR.draw_triples(acc, wn, rank, excl, acc_ref=0.3)
  assert a == UR.draw_triples(acc, wn, rank, excl, acc_ref=0.3) and len(a) == UR.N_TRIPLES
  used = set()
  for c, t in a:
    assert acc[c] >= 0.4 and abs(acc[t] - 0.3) <= 0.1 and wn[c, t] >= 10 and min(rank[c, t], rank[t, c]) > 100
    assert c not in excl | used and t not in excl | used
    used |= {c, t}
  A, B = UR.far_pair(acc, wn, rank, a[0][1], a[0][0], [])
  assert A != B and min(wn[A, a[0][1]], wn[A, a[0][0]]) >= 10
  rows = [{'photo': 'val:1', 'top1': 3, 'p_target': 0.2, 'p_source': 0.45},
          {'photo': 'val:2', 'top1': 3, 'p_target': 0.005, 'p_source': 0.9},
          {'photo': 'val:3', 'top1': 4, 'p_target': 0.0, 'p_source': 0.45}]
  ok, rule = UR.step1_rule(rows, 3)
  assert [r['photo'] for r in ok] == ['val:2'] and 'p_target' in rule
  ok, rule = UR.step1_rule(rows[:1], 3)
  assert [r['photo'] for r in ok] == ['val:1'] and rule == 'top1 = source'


def test_order_j_prefers_photos_passing_in_both():
  mk = lambda ps: [{'photo': p} for p in ps]
  sel_r = {'arms': {'strong_source': {'source': 7, 'photos': mk(['val:1', 'val:2', 'val:3']),
                                      'reserve': mk(['val:4', 'val:5'])}}}
  sel_e = {'arms': {'s': {'source': 7, 'photos': mk(['val:5', 'val:2']), 'reserve': mk(['val:9'])}}}
  out = UR.order_j(sel_r, sel_e)['arms']['strong_source']
  assert [r['photo'] for r in out['photos']] == ['val:2', 'val:5', 'val:1'] and out['n_in_both'] == 2
  assert [r['photo'] for r in out['reserve']] == ['val:3', 'val:4']


def test_lgo_variants_share_the_base_slots():
  labels = np.repeat(np.arange(10), 40)[np.random.default_rng(0).permutation(400)]
  by = UR.class_index(labels)
  rest = [0, 3, 4, 5, 6, 7, 8, 9]
  base = UR.lgo_indices(by, rest, None, seed=1, step=17)
  assert np.array_equal(base, UR.lgo_indices(by, rest, None, seed=1, step=17)) and len(base) == 255
  assert set(labels[base]) <= set(rest)
  for c in (1, 2, 3):
    v = UR.lgo_indices(by, rest, c, seed=1, step=17)
    assert np.array_equal(v[:223], base[:223]) and (labels[v[223:]] == c).all()
  assert not np.array_equal(base, UR.lgo_indices(by, rest, None, seed=1, step=18))


def test_frozen_v_continuation_repeats_the_frozen_run():
  """A frozen-v run (as in the fine-tuning) continued from its snapshot at step 3 with frozen_v_continuation gives the
  same weights as the uninterrupted run (the bias correction of v stays at the freeze step)."""
  torch.manual_seed(0)
  base = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3)).double()
  g = torch.Generator().manual_seed(1)
  x, y = torch.randn(8, 6, 5, generator=g, dtype=DT), torch.randn(8, 6, 3, generator=g, dtype=DT)
  ref = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3)).double()
  ref.load_state_dict(base.state_dict())
  opt = torch.optim.AdamW(ref.parameters(), lr=1e-2)
  for t in range(3):                                           # pretraining by torch, t_freeze = 3
    opt.zero_grad()
    ((ref(x[t]) - y[t]) ** 2).mean().backward()
    opt.step()
  snap0 = snapshot_from_torch_adamw(opt, {'l0': ref[0], 'l2': ref[2]})
  m1 = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3)).double()
  m1.load_state_dict(ref.state_dict())
  at = GroupAttribution({'l0': m1[0], 'l2': m1[2]}, n_groups=3, optimizer='adamw', lr=1e-2, weight_decay=0.05, dtype=DT)
  at.load_snapshot(snap0, history_group=2)
  at.freeze_second_moment()
  for t in range(3, 8):
    ((m1(x[t]) - y[t]) ** 2).mean().backward()
    at.step(torch.tensor([0, 1, 0, 1, 0, 1]))
    if t == 4:
      mid = ({k: v.clone() for k, v in m1.state_dict().items()}, at.snapshot())
  m2 = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3)).double()
  m2.load_state_dict(mid[0])
  cont = UR.frozen_v_continuation({'l0': m2[0], 'l2': m2[2]}, mid[1], t_freeze=3, lr=1e-2, wd=0.05, dtype=DT)
  for t in range(5, 8):
    ((m2(x[t]) - y[t]) ** 2).mean().backward()
    cont.step(torch.zeros(6, dtype=torch.long))
  for p1, p2 in zip(m1.parameters(), m2.parameters()):
    assert torch.allclose(p1, p2, atol=1e-13, rtol=0)
  assert UR.full_saves([], 500) == []


def test_cli_run_and_decide(toy, tmp_path, monkeypatch, capsys):
  """The run and decide subcommands on the toy (model, photo and selection replaced by the toy's), and the queue
  helpers of the arms module."""
  import sys
  import experiments.convnext_umbrella as U
  e1dir = tmp_path / 'e1'
  e1dir.mkdir()
  for name, key in (('atlas_25020.pt', 'atlas'), ('attr_25020.pt', 'attr'), ('attr_init.pt', 'init'),
                    ('model_25020.pt', 'state')):
    torch.save(toy['e1'][key], e1dir / name)
  sel = tmp_path / 'selection.json'
  json.dump({'arms': {'toy': {'target': T, 'source': S, 'rel_target': RELB, 'rel_source': RELU, 'A': A_, 'B': B_,
                              'photos': [{'photo': 'val:3'}, {'photo': 'val:4'}]}}}, open(sel, 'w'))
  monkeypatch.setattr(UA, 'build_convnext', lambda size: build_convnext('tiny', n_classes=K))
  monkeypatch.setattr(U, 'photo_tensor', lambda spec, val, dev, train: photo_x())
  out = tmp_path / 'an' / 'analysis.json'
  out.parent.mkdir()
  monkeypatch.setattr(sys, 'argv', ['x', 'run', '--run', toy['runs']['taught'], '--untaught', toy['runs']['untaught'],
                                    '--tag', 'B', '--val', 'none', '--sel', str(sel), '--e1', str(e1dir), '--device',
                                    'cpu', '--skip', 'M2', '--perm', '100', '--out', str(out)])
  UA.main()
  rec = json.load(open(out))
  assert rec['tag'] == 'B' and rec['kind'] == 'photo' and rec['source'] == S and rec['tau'] == TAU
  assert 'M1' in rec and 'M3' in rec and 'M4' in rec and 'M2' not in rec
  assert rec['M3']['poles']['n_B_pole'] > 0 and os.path.exists(str(out)[:-5] + '.pt')
  monkeypatch.setattr(sys, 'argv', ['x', 'decide', '--root', str(tmp_path), '--out', str(tmp_path / 'dec.json')])
  UA.main()
  dec = json.load(open(tmp_path / 'dec.json'))
  assert dec['n']['B'] == 1 and dec['P-C1a']['per_run'][0]['photo'] == 'toy:0'
  capsys.readouterr()
  monkeypatch.setattr(sys, 'argv', ['x', 'photos', '--sel', str(sel), '--arm', 'toy'])
  UR.main()
  assert capsys.readouterr().out.split() == ['val:3', 'val:4']
  monkeypatch.setattr(sys, 'argv', ['x', 'full_saves', toy['runs']['taught'], toy['runs']['solo'], '--phase2', '500'])
  UR.main()
  assert capsys.readouterr().out.strip() == f'{TAU},{TAU + 500}'
  json.dump({'lr_star': 0.001}, open(tmp_path / 'lr.json', 'w'))
  monkeypatch.setattr(sys, 'argv', ['x', 'lr_star', '--file', str(tmp_path / 'lr.json')])
  UR.main()
  assert capsys.readouterr().out.strip() == '0.001'


def test_vsets_on_a_fake_validation_set(tmp_path):
  """cmd_vsets and load_vsets on a 2-file validation parquet of 6 classes: sets, exclusions, images, and the shared
  first-10 cache rebuilt after it is lost."""
  pytest.importorskip('pyarrow')
  import io
  import types
  import pyarrow as pa
  import pyarrow.parquet as pq
  from PIL import Image
  from experiments.convnext_umbrella import val_images
  rng = np.random.default_rng(0)
  labels = np.repeat(np.arange(6), 13)[rng.permutation(78)]
  d = tmp_path / 'val' / 'data'
  d.mkdir(parents=True)
  for f, part in enumerate(np.array_split(np.arange(78), 2)):
    rows = []
    for i in part:
      buf = io.BytesIO()
      Image.new('RGB', (60 + i % 7, 50), (int(labels[i]) * 40, int(i) % 256, 7)).save(buf, format='JPEG')
      rows.append({'image': {'bytes': buf.getvalue(), 'path': None}, 'label': int(labels[i])})
    pq.write_table(pa.Table.from_pylist(rows), d / f'validation-{f:05d}-of-00002.parquet')
  first = lambda c, k: [int(g) for g in np.flatnonzero(labels == c)[:k]]
  arm = {'target': 1, 'source': 2, 'rel_target': [3], 'rel_source': [], 'A': 0, 'B': 4,
         'excluded_from_eval': first(1, 8) + first(2, 8) + [int(np.flatnonzero(labels == 2)[9])],
         'photos': [{'photo': f'val:{int(np.flatnonzero(labels == 2)[9])}'}], 'reserve': []}
  armE = dict(arm, target=4, source=5, excluded_from_eval=first(4, 8) + first(5, 8) + [first(0, 3)[2]],
              photos=[{'photo': f'val:{first(0, 3)[2]}'}])
  sel, selE = tmp_path / 'sel.json', tmp_path / 'selE.json'
  json.dump({'arms': {'strong_source': arm}}, open(sel, 'w'))
  json.dump({'arms': {'E0': armE}}, open(selE, 'w'))
  out, f10 = tmp_path / 'vsets.pt', tmp_path / 'cache' / 'first10.pt'
  UA.cmd_vsets(types.SimpleNamespace(sel=[str(sel), str(selE)], arm='strong_source', val=str(tmp_path / 'val'),
                                     first10=str(f10), out=str(out)))
  vs = UA.load_vsets(str(out))
  assert sorted(set(labels[vs['V_ref']['idx']])) == [0, 4, 5]
  assert first(0, 3)[2] not in vs['V_ref']['idx'] and len(vs['V_ref']['idx']) == 29        # the arm-E photo is out
  assert set(first(4, 8) + first(5, 8)) <= set(vs['V_ref']['idx'])                          # other arms' probe_8 stay
  assert len(vs['V_umb']['idx']) == 13 - 9 and len(vs['V_bee']['idx']) == 13 - 8
  assert torch.equal(vs['V_ref']['x'], val_images(str(tmp_path / 'val'), vs['V_ref']['idx']))
  assert vs['V_umb']['x'].dtype == torch.uint8 and vs['V_umb']['x'].shape[1:] == (3, 224, 224)
  os.remove(f10)
  assert torch.equal(UA.load_vsets(str(out))['V_ref']['x'], vs['V_ref']['x'])


def test_verify_helpers():
  """The namespace for convnext_umbrella.branch comes from the fine-tuning script's own parser (same branch key and
  defaults), and a photo that fails at the branch point is replaced in its slot by the next passing reserve photo."""
  a = UR.umbrella_args(['--', '--sel', 's.json', '--arm', 'strong_source', '--snapshot', 'snap.pt', '--val', 'v',
                        '--branch_dir', 'b', '--mode', 'untaught', '--lr', '1e-3', '--seed', '2'])
  assert (a.arm, a.seed, a.lr, a.replay, a.rest, a.rest_exclude, a.warmup, a.warmup_ramp, a.wd) == \
         ('strong_source', 2, 1e-3, 'T:32,S:32,A:32,B:32', 127, '', 300, 100, 0.05)
  ok = lambda p, o: {'photo': p, 'ok': o}
  specs, swaps = UR.replace_photos([ok('a', True), ok('b', False), ok('c', True), ok('d', False)],
                                   [ok('r1', False), ok('r2', True)], 4)
  assert specs == ['a', 'r2', 'c'] and swaps == [('b', 'r2'), ('d', None)]
