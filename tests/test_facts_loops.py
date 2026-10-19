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
