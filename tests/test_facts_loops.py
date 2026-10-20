"""rcdl.facts looping: K = 1 is the plain model, K passes reuse the same block, the noise and Jacobian probes."""

import math

import pytest
import torch
import torch.nn.functional as F

from rcdl import ckpt
from rcdl.facts import FactsConfig, FactsModel, loop_probe, run_facts, tangent_spectrum


def _cfg(**kw):
  base = dict(n_people=32, n_attrs=2, n_values=8, d=16, heads=2, layers=2, width=24, name_tokens=2, name_vocab=16,
              n_filler=3, random_name_pos=True, steps=10, eval_every=5, batch_size=8)
  return FactsConfig(**(base | kw))


def _model(cfg, seed=0):
  torch.manual_seed(seed)
  m = FactsModel(cfg, torch.Generator().manual_seed(seed))
  for p in m.parameters():                     # move away from the zero-initialised cos readout
    p.data.add_(0.1 * torch.randn_like(p))
  return m.eval()


def _tokens(cfg, B=6, seed=1):
  from rcdl.facts import name_region
  V = name_region(cfg) + cfg.n_attrs + cfg.filler_vocab
  return torch.randint(0, V, (B, cfg.name_tokens + cfg.n_filler + 1), generator=torch.Generator().manual_seed(seed))


def _reference(m, tokens):
  """The forward pass before looping existed (one pass of every FFN)."""
  h = m.emb(tokens) + m.pos
  for i in range(m.cfg.layers):
    a = m.ln1[i](h)
    h = h + m.attn[i](a, a, a, attn_mask=m.mask, need_weights=False)[0]
    if i == m.cfg.layers - 1:
      h = h[:, -1:]
    h = h + m.ffn[i](m.ln2[i](h))
  return m.head(m.ln_f(h[:, -1]))


@pytest.mark.parametrize('ffn', ['geglu', 'cos'])
def test_one_pass_is_the_plain_model_and_k_passes_reuse_the_block(ffn):
  cfg = _cfg(ffn=ffn, cos_sigma=2.0)
  m, x = _model(cfg), _tokens(cfg)
  with torch.no_grad():
    torch.testing.assert_close(m(x), _reference(m, x), rtol=0, atol=0)
    i, a = cfg.layers - 1, 0.5
    m2 = _model(_cfg(ffn=ffn, cos_sigma=2.0, loops=3, loop_alpha=a))
    m2.load_state_dict(m.state_dict())
    h = m2.last_state(x)
    for _ in range(3):
      h = h + a * m2.ffn[i](m2.ln2[i](h))
    torch.testing.assert_close(m2(x), m2.head(m2.ln_f(h[:, -1])))
    torch.testing.assert_close(m2(x, loops=1), m2.head(m2.ln_f((m2.last_state(x) + a * m2.block(i, m2.last_state(x)))[:, -1])))
    assert not torch.allclose(m2(x, loops=1), m2(x))


def test_noise_has_the_requested_relative_size_and_zero_noise_changes_nothing():
  cfg = _cfg(ffn='geglu')
  m, x = _model(cfg), _tokens(cfg)
  with torch.no_grad():
    torch.testing.assert_close(m(x, noise=0.0), m(x), rtol=0, atol=0)
    h = m.last_state(x)
    g = torch.Generator().manual_seed(3)
    z = torch.randn(h.shape, generator=g)
    noisy = h + 0.25 * h.norm(dim=-1, keepdim=True) * z / z.norm(dim=-1, keepdim=True)
    assert torch.allclose((noisy - h).norm(dim=-1) / h.norm(dim=-1), torch.full(h.shape[:2], 0.25), atol=1e-5)
    torch.testing.assert_close(m(x, noise=0.25, noise_gen=torch.Generator().manual_seed(3)),
                               m.head(m.ln_f(m.loop(noisy)[:, -1])))


@pytest.mark.parametrize('ffn', ['geglu', 'cos'])
def test_jacobian_probe_matches_finite_differences(ffn):
  cfg = _cfg(ffn=ffn, cos_sigma=2.0, loops=2, loop_alpha=0.7)
  m, x = _model(cfg), _tokens(cfg, B=4)
  y = torch.randint(0, cfg.n_values, (len(x),))
  out = loop_probe(m.double(), x, y, n_jac=4)
  i, a = cfg.layers - 1, cfg.loop_alpha
  step = lambda v: v + a * m.block(i, v[None, None])[0, 0]
  with torch.no_grad():
    h = m.last_state(x)[:, 0]
    rads, steps = [], []
    for v in h:
      J = torch.stack([(step(v + 1e-6 * e) - step(v - 1e-6 * e)) / 2e-6 for e in torch.eye(len(v), dtype=v.dtype)], 1)
      rads.append(torch.linalg.eigvals(J).abs().max())
      steps.append((step(v) - v).norm() / v.norm())
  assert out['jacobian_by_pass'][0]['spectral_radius_median'] == pytest.approx(float(torch.stack(rads).median()), rel=1e-5)
  assert out['jacobian_by_pass'][0]['rel_step_median'] == pytest.approx(float(torch.stack(steps).median()), rel=1e-6)
  assert len(out['jacobian_by_pass']) == 2 and set(out['acc_by_K']) == {'1', '2', '4', '8', '16'}
  assert set(out['acc_by_noise']) == set(out['acc_by_noise_end']) == {'0.1', '0.25', '0.5', '1.0'}


def test_rbf_cannot_loop():
  with pytest.raises(ValueError):
    FactsModel(_cfg(ffn='rbf', loops=2), torch.Generator().manual_seed(0))


def test_tiny_run_reports_the_probe():
  r = run_facts(_cfg(ffn='cos', cos_sigma=2.0, loops=2, final_probe=True, steps=6, eval_every=3))
  p = r['loop_probe']
  assert 0 <= p['acc_by_K']['2'] <= 1 and len(p['jacobian_by_pass']) == 2
  assert math.isfinite(p['jacobian_by_pass'][1]['spectral_norm_median'])
  assert 'loop_probe' not in run_facts(_cfg(ffn='geglu', steps=4, eval_every=2))


@pytest.mark.parametrize('ffn', ['geglu', 'cos'])
def test_untied_control_uses_its_own_block_per_pass_and_keeps_the_shared_model(ffn):
  torch.manual_seed(0)
  tied = FactsModel(_cfg(ffn=ffn, cos_sigma=2.0, loops=3), torch.Generator().manual_seed(0))
  torch.manual_seed(0)
  m = FactsModel(_cfg(ffn=ffn, cos_sigma=2.0, loops=3, loop_tied=False), torch.Generator().manual_seed(0))
  sd = m.state_dict()
  for k, v in tied.state_dict().items():                 # shared modules identical to the tied model
    torch.testing.assert_close(sd[k], v, rtol=0, atol=0)
  assert len(m.extra_ffn) == 2 and not hasattr(tied, 'extra_ffn')
  with torch.no_grad():
    for p in m.parameters():
      p.add_(0.1 * torch.randn_like(p))
  m.eval()
  x, i = _tokens(m.cfg), m.cfg.layers - 1
  with torch.no_grad():
    h = m.last_state(x)
    h = h + m.ffn[i](m.ln2[i](h))
    for j in range(2):
      h = h + m.extra_ffn[j](m.extra_ln[j](h))
    torch.testing.assert_close(m(x), m.head(m.ln_f(h[:, -1])))
    with pytest.raises(ValueError):
      m(x, loops=2)
  y = torch.randint(0, m.cfg.n_values, (len(x),))
  out = loop_probe(m.double(), x, y, n_jac=3)
  assert set(out['acc_by_K']) == {'3'} and len(out['jacobian_by_pass']) == 3
  norms = []
  for v in m.last_state(x[:3])[:, 0].detach():           # pass 2 runs the first extra block at the state after pass 1
    v = v + m.pass_block(0, v[None, None])[0, 0]
    J = torch.autograd.functional.jacobian(lambda u: u + m.pass_block(1, u[None, None])[0, 0], v.detach())
    norms.append(torch.linalg.matrix_norm(J, ord=2))
  assert out['jacobian_by_pass'][1]['spectral_norm_median'] == pytest.approx(float(torch.stack(norms).median()), rel=1e-6)



@pytest.mark.parametrize('ffn', ['geglu', 'cos'])
def test_tangent_spectrum_drops_the_two_trivial_unit_eigenvalues(ffn):
  m = _model(_cfg(ffn=ffn, cos_sigma=2.0)).double()
  i = m.cfg.layers - 1
  h = m.last_state(_tokens(m.cfg, B=3))[:, 0].detach()
  step = lambda v: v + 0.7 * m.block(i, v[None, None])[0, 0]
  J = torch.stack([torch.autograd.functional.jacobian(step, v) for v in h])
  for v, Jv in zip(h, J):            # LayerNorm ignores scale and shift: J h = h, J 1 = 1 (up to its eps = 1e-5)
    torch.testing.assert_close(Jv @ v, v, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(Jv @ torch.ones_like(v), torch.ones_like(v), rtol=1e-6, atol=1e-6)
  tan = tangent_spectrum(J, h)
  assert tan.shape == (3, h.shape[1] - 2)
  for Jv, t in zip(J, tan):
    ev = torch.linalg.eigvals(Jv)
    drop = (ev - 1).abs().argsort()[:2]                     # the two trivial eigenvalues
    keep = torch.ones(len(ev), dtype=torch.bool)
    keep[drop] = False
    torch.testing.assert_close(ev[keep].abs().sort().values, t, rtol=1e-4, atol=1e-6)


def test_probe_v2_measures_agree_with_direct_computation():
  cfg = _cfg(ffn='cos', cos_sigma=2.0, loops=2, loop_alpha=0.8, eval_facts=0)
  m = _model(cfg)
  x = _tokens(cfg, B=40, seed=4)
  y = torch.randint(0, cfg.n_values, (len(x),), generator=torch.Generator().manual_seed(2))
  out = loop_probe(m, x, y, n_jac=4, seed=5, k_max=8)
  acc = lambda logits: float((logits.argmax(-1) == y).float().mean())
  with torch.no_grad():
    for e in ('0.1', '0.5'):                                 # propagated noise: the same draws as forward(noise=e)
      assert out['acc_by_noise'][e] == pytest.approx(acc(m(x, noise=float(e), noise_gen=torch.Generator().manual_seed(5 + int(float(e) * 1000)))))
    g = torch.Generator().manual_seed(5 + 250)
    h = m.last_state(x)
    z = torch.randn(h.shape, generator=g)
    n = 0.25 * h.norm(dim=-1, keepdim=True) * z / z.norm(dim=-1, keepdim=True)
    assert out['acc_by_noise_end']['0.25'] == pytest.approx(acc(m.head(m.ln_f((m.loop(h) + n)[:, -1]))))
    assert out['acc_by_alpha']['1.5'] == pytest.approx(acc(m(x, alpha=1.5)))
    traj = {r['pass']: r for r in out['trajectory']}
    for k in (1, 2, 4, 8):
      assert traj[k]['acc'] == pytest.approx(out['acc_by_K'][str(k)])
    h1, h2 = m.loop(h, 1)[:, 0], m.loop(h, 2)[:, 0]
    s1, s2 = h1 - h[:, 0], h2 - h1
    assert traj[2]['step_cos_median'] == pytest.approx(float(torch.nn.functional.cosine_similarity(s1, s2, dim=-1).median()), rel=1e-5)
    assert traj[2]['flips'] == pytest.approx(float((m.head(m.ln_f(h2)).argmax(-1) != m.head(m.ln_f(h1)).argmax(-1)).float().mean()))
  p = out['jacobian_by_pass'][0]
  assert 1 <= p['spectral_radius_median'] and 0 <= p['tangent_contracting_share_median'] <= 1
  assert out['jacobian_noisy']['pass'] == 1 and out['probe_version'] == 2


def test_run_saves_a_model_that_reproduces_the_outputs(tmp_path):
  cfg = _cfg(ffn='cos', cos_sigma=2.0, loops=2, steps=6, eval_every=3)
  r = run_facts(cfg, model_path=str(tmp_path / 'model_t'))
  st = ckpt.load(str(tmp_path / 'model_t'))
  assert st['cfg'] == r['cfg']
  m = FactsModel(FactsConfig(**st['cfg']), torch.Generator().manual_seed(99))  # other random features
  assert not torch.equal(m.ffn[1].W, st['model']['ffn.1.W'])
  m.load_state_dict(st['model'])                          # the fixed random features are buffers, so they are saved
  assert torch.equal(m.ffn[1].W, st['model']['ffn.1.W']) and sorted(st['model']) == sorted(m.state_dict())


def test_ffn_variants_have_the_intended_trained_and_frozen_parts():
  base = dict(n_people=32, n_attrs=2, n_values=8, d=16, heads=2, layers=2, width=24, name_tokens=2, name_vocab=16,
              n_filler=3, random_name_pos=True)
  count = lambda f: (sum(p.numel() for p in f.parameters() if p.requires_grad), sum(b.numel() for b in f.buffers()))
  g = lambda: torch.Generator().manual_seed(0)
  m = FactsModel(FactsConfig(**base, ffn='relu'), g())
  assert count(m.ffn[0]) == (16 * 24 + 24 + 24 * 16 + 16, 0)
  m = FactsModel(FactsConfig(**base, ffn='rfrelu'), g())
  assert count(m.ffn[0]) == (24 * 16 + 16, 24 * 16) and torch.all(m.ffn[0].o.weight == 0)
  fixed = FactsModel(FactsConfig(**base, ffn='cos', cos_sigma=2.0), g())
  trained = FactsModel(FactsConfig(**base, ffn='cos', cos_sigma=2.0, cos_train=True), g())
  assert count(fixed.ffn[0]) == (24 * 16 + 16, 24 * 16 + 24)
  assert count(trained.ffn[0]) == (24 * 16 + 24 + 24 * 16 + 16, 0)
  torch.testing.assert_close(trained.ffn[0].W, fixed.ffn[0].W, rtol=0, atol=0)   # same draw, now trained
  m = FactsModel(FactsConfig(**base, ffn='rbf', rbf_layers='last', steps=2), g())
  assert isinstance(m.ffn[0], torch.nn.Module) and type(m.ffn[0]).__name__ == 'GeGLU'
  for ffn in ('relu', 'rfrelu'):                    # the loop, the untied control and the probe work for the new kinds
    for tied in (True, False):
      r = run_facts(_cfg(ffn=ffn, loops=2, loop_tied=tied, final_probe=True, steps=4, eval_every=2))
      assert len(r['loop_probe']['jacobian_by_pass']) == 2
  r = run_facts(_cfg(ffn='cos', cos_sigma=2.0, cos_train=True, loops=2, final_probe=True, steps=4, eval_every=2))
  assert math.isfinite(r['loop_probe']['jacobian_by_pass'][0]['tangent_radius_median'])


@pytest.mark.parametrize('ffn', ['geglu', 'relu'])
def test_zero_init_only_zeroes_the_output_weight(ffn):
  torch.manual_seed(0)
  a = FactsModel(_cfg(ffn=ffn, loops=2), torch.Generator().manual_seed(0))
  torch.manual_seed(0)
  b = FactsModel(_cfg(ffn=ffn, loops=2, ffn_zero_init=True), torch.Generator().manual_seed(0))
  sa, sb = a.state_dict(), b.state_dict()
  for k in sa:                                      # every other parameter is the same draw
    if k.endswith('.o.weight') and k.startswith('ffn.'):
      assert torch.all(sb[k] == 0) and not torch.all(sa[k] == 0)
    else:
      torch.testing.assert_close(sa[k], sb[k], rtol=0, atol=0)
  b.eval()
  x = _tokens(b.cfg)
  with torch.no_grad():                             # at initialisation the looped block writes only its output bias
    h = b.last_state(x)
    i = b.cfg.layers - 1
    torch.testing.assert_close(b.block(i, h), b.ffn[i].o.bias.expand_as(h))


def test_infiloop_streaming_equals_the_explicit_weighted_average():
  from rcdl.facts import infiloop
  torch.manual_seed(0)
  d, K = 6, 5
  h0 = torch.randn(3, d, dtype=torch.float64)
  W = torch.randn(d, d, dtype=torch.float64) * 0.3
  cand = lambda k, v: v + torch.tanh(v @ W) * (1 + 0.1 * k)
  q, r = torch.randn(d, dtype=torch.float64), torch.tensor(0.3, dtype=torch.float64)
  out = infiloop(h0, K, cand, q, r)
  lam = torch.nn.functional.softplus(r)
  s = lambda z: (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6) @ q).clamp(-8, 8)
  zs, h = [h0], h0
  for k in range(K):                                 # eq. 2: h_t = sum_i alpha_t,i z_i, alpha ~ exp(s_i - lam (t - i))
    zs.append(cand(k, h))
    t = len(zs) - 1
    w = torch.stack([torch.exp(s(z) - lam * (t - i)) for i, z in enumerate(zs)])        # (t + 1, n)
    h = (w[..., None] * torch.stack(zs)).sum(0) / w.sum(0)[..., None]
  torch.testing.assert_close(out, h, rtol=1e-5, atol=1e-6)    # infiloop keeps scores and sums in fp32 by design


def test_infiloop_model_starts_at_equal_weights_and_probe_runs():
  cfg = _cfg(ffn='cos', cos_sigma=2.0, loops=2, loop_mode='infiloop')
  m = _model(cfg)
  with torch.no_grad():
    m.loop_q.zero_()
    m.loop_r.fill_(math.log(math.expm1(-math.log(0.55))))
    x = _tokens(cfg)
    h0 = m.last_state(x)
    i = cfg.layers - 1
    z1 = h0 + m.block(i, h0)
    h1 = (0.55 * h0 + z1) / 1.55                      # q = 0: every score is 1, b = 0.55
    z2 = h1 + m.block(i, h1)
    h2 = (0.55 * (0.55 * h0 + z1) + z2) / (0.55 * 1.55 + 1)
    torch.testing.assert_close(m.loop(h0), h2)
  y = torch.randint(0, cfg.n_values, (len(x),))
  out = loop_probe(m, x, y, n_jac=2, k_max=4)
  assert out['jacobian_by_pass'] == [] and len(out['trajectory']) == 4
  assert out['trajectory'][1]['acc'] == pytest.approx(out['acc_by_K']['2'])
  with pytest.raises(ValueError):
    FactsModel(_cfg(loop_mode='gated'), torch.Generator().manual_seed(0))


def test_cosine_schedule_reaches_zero_at_the_end(monkeypatch):
  lrs = []
  real = torch.optim.Adam.step
  def spy(self, *a, **k):
    lrs.append(self.param_groups[0]['lr'])
    return real(self, *a, **k)
  monkeypatch.setattr(torch.optim.Adam, 'step', spy)
  run_facts(_cfg(steps=8, eval_every=4, lr=1e-3, lr_schedule='cosine'))
  assert lrs[0] == pytest.approx(1e-3) and lrs[4] == pytest.approx(0.5e-3)
  assert lrs[-1] == pytest.approx(0.5e-3 * (1 + math.cos(math.pi * 7 / 8))) and lrs == sorted(lrs, reverse=True)


@pytest.mark.parametrize('mode', ['residual', 'infiloop'])
def test_all_passes_reads_the_head_after_each_pass(mode):
  cfg = _cfg(ffn='cos', cos_sigma=2.0, loops=3, loop_mode=mode)
  m, x = _model(cfg), _tokens(cfg)
  with torch.no_grad():
    if mode == 'infiloop':
      m.loop_q.normal_()                             # an active gate
    per = m(x, all_passes=True)
    assert per.shape == (3, len(x), cfg.n_values)
    torch.testing.assert_close(per[-1], m(x), rtol=0, atol=0)
    h0 = m.last_state(x)
    for k in (1, 2):                                 # the state after pass k is the state of a k-pass loop (streaming)
      torch.testing.assert_close(per[k - 1], m.head(m.ln_f(m.loop(h0, k)[:, -1])))


def test_deep_supervision_trains_on_the_mean_loss_over_passes_and_is_inert_at_one_pass(monkeypatch):
  import rcdl.facts as facts_mod
  seen = []
  real = facts_mod.F.cross_entropy
  def spy(logits, target, *a, **k):
    seen.append((tuple(logits.shape), tuple(target.shape)))
    return real(logits, target, *a, **k)
  monkeypatch.setattr(facts_mod.F, 'cross_entropy', spy)
  run_facts(_cfg(ffn='cos', cos_sigma=2.0, loops=3, deep_supervision=True, steps=2, eval_every=10))
  train = [s for s in seen if s[0][0] == 3 * 8]      # batch 8, K = 3: (K B, n_values) against (K B,)
  assert train and all(t == (24,) for _, t in train)
  monkeypatch.setattr(facts_mod.F, 'cross_entropy', real)
  a = run_facts(_cfg(ffn='cos', cos_sigma=2.0, steps=6, eval_every=3))
  b = run_facts(_cfg(ffn='cos', cos_sigma=2.0, steps=6, eval_every=3, deep_supervision=True))
  strip = lambda log: [{k: v for k, v in r.items() if k != 'sec'} for r in log]
  assert strip(a['log']) == strip(b['log'])          # K = 1: deep supervision is the plain loss


def test_deep_supervision_loss_equals_the_mean_of_per_pass_losses():
  cfg = _cfg(ffn='geglu', loops=3)
  m, x = _model(cfg), _tokens(cfg, B=10)
  y = torch.randint(0, cfg.n_values, (len(x),), generator=torch.Generator().manual_seed(3))
  with torch.no_grad():
    per = m(x, all_passes=True)
    flat = F.cross_entropy(per.flatten(0, 1), y.repeat(3))
    torch.testing.assert_close(flat, torch.stack([F.cross_entropy(per[k], y) for k in range(3)]).mean())


@pytest.mark.parametrize('ffn', ['geglu', 'cos'])
def test_train_probe_leaves_training_unchanged_and_matches_direct_computation(ffn):
  base = dict(ffn=ffn, cos_sigma=2.0, loops=3, steps=6, eval_every=3, eval_facts=0)
  a, b = run_facts(_cfg(**base)), run_facts(_cfg(**base, train_probe=True))
  strip = lambda log: [{k: v for k, v in r.items() if k not in ('sec', 'pass_probe')} for r in log]
  assert strip(a['log']) == strip(b['log'])
  p = b['log'][-1]['pass_probe']
  assert len(p['loss']) == len(p['acc']) == 4 and len(p['rel_step']) == len(p['descent_cos']) == 3
  assert len(p['input_shift']) == len(p['overlap_prev']) == len(p['overlap_first']) == 2
  assert p['acc'][-1] == pytest.approx(b['log'][-1]['acc'])      # 64 facts < 1,024: the probe sees every fact


def test_pass_probe_measures_agree_with_direct_computation():
  from rcdl.facts import pass_probe
  cfg = _cfg(ffn='cos', cos_sigma=2.0, loops=2, loop_alpha=0.7)
  m, x = _model(cfg), _tokens(cfg, B=30, seed=6)
  y = torch.randint(0, cfg.n_values, (len(x),), generator=torch.Generator().manual_seed(4))
  p = pass_probe(m, x, y)
  i = cfg.layers - 1
  h0 = m.last_state(x)[:, -1].detach()
  with torch.no_grad():
    h1 = h0 + 0.7 * m.ffn[i](m.ln2[i](h0))
    h2 = h1 + 0.7 * m.ffn[i](m.ln2[i](h1))
  v = h0.clone().requires_grad_(True)
  loss0 = F.cross_entropy(m.head(m.ln_f(v)), y, reduction='sum')
  g0, = torch.autograd.grad(loss0, v)
  assert p["loss"][0] == pytest.approx(float(loss0.detach()) / len(y), rel=1e-5)
  assert p['descent_cos'][0] == pytest.approx(float(F.cosine_similarity(h1 - h0, -g0, dim=-1).median()), rel=1e-4)
  assert p['rel_step'][1] == pytest.approx(float(((h2 - h1).norm(dim=-1) / h1.norm(dim=-1)).median()), rel=1e-5)
  with torch.no_grad():
    x0, x1 = m.ln2[i](h0), m.ln2[i](h1)
    phi = lambda u: math.sqrt(2 / cfg.width) * torch.cos(u @ m.ffn[i].W.T + m.ffn[i].b)
    assert p['input_shift'][0] == pytest.approx(float((x1 - x0).norm(dim=-1).median()), rel=1e-5)
    assert p['overlap_prev'][0] == pytest.approx(float(F.cosine_similarity(phi(x1), phi(x0), dim=-1).median()), rel=1e-5)
  assert m.training                                  # the probe runs during training and leaves the model in train mode


def test_gate_logging_weights_and_the_gate_off_probe():
  from rcdl.facts import pass_probe
  r = run_facts(_cfg(ffn='cos', cos_sigma=2.0, loops=2, loop_mode='infiloop', steps=4, eval_every=2, train_probe=True,
                     final_probe=True, lr=1e-2))
  first, last = r['log'][0], r['log'][-1]
  assert first['gate_q_norm'] == 0 and first['gate_beta'] == pytest.approx(0.55)
  assert last['gate_q_norm'] > 0                     # q trains
  g = last['pass_probe']['gate']
  assert sum(g['weight_median']) == pytest.approx(1, abs=0.2) and len(g['logit_sd']) == 3
  assert first['pass_probe']['gate']['logit_sd'] == [0.0, 0.0, 0.0]   # q = 0: every score is 0, weights b^(K-t) / sum
  w0 = first['pass_probe']['gate']['weight_median']
  torch.testing.assert_close(torch.tensor(w0), torch.tensor([0.55 ** 2, 0.55, 1.0]) / (0.55 ** 2 + 0.55 + 1))
  probe = r['loop_probe']
  assert set(probe['acc_by_K_q0']) == {'1', '2', '4', '8', '16'} and probe['gate']['q_norm'] == pytest.approx(last['gate_q_norm'])
  cfg = _cfg(ffn='cos', cos_sigma=2.0, loops=2, loop_mode='infiloop')
  m, x = _model(cfg), _tokens(cfg, B=20)
  y = torch.randint(0, cfg.n_values, (len(x),))
  with torch.no_grad():
    m.loop_q.normal_()
    q = m.loop_q.clone()
  out = loop_probe(m, x, y, n_jac=2, k_max=2)
  assert torch.equal(m.loop_q, q)                    # the gate-off probe restores q
  with torch.no_grad():
    m.loop_q.zero_()
    assert out['acc_by_K_q0']['2'] == pytest.approx(float((m(x).argmax(-1) == y).float().mean()))
  pp = pass_probe(m, x, y)
  assert pp['gate']['q_norm'] == 0
