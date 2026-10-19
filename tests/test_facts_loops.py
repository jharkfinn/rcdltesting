"""rcdl.facts looping: K = 1 is the plain model, K passes reuse the same block, the noise and Jacobian probes."""

import math

import pytest
import torch
import torch.nn.functional as F

from rcdl.facts import FactsConfig, FactsModel, loop_probe, run_facts


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
  assert len(out['jacobian_by_pass']) == 2 and set(out['acc_by_K']) == {'1', '2', '4', '8'}
  assert set(out['acc_by_noise']) == {'0.1', '0.25', '0.5'}


def test_rbf_cannot_loop():
  with pytest.raises(ValueError):
    FactsModel(_cfg(ffn='rbf', loops=2), torch.Generator().manual_seed(0))


def test_tiny_run_reports_the_probe():
  r = run_facts(_cfg(ffn='cos', cos_sigma=2.0, loops=2, final_probe=True, steps=6, eval_every=3))
  p = r['loop_probe']
  assert 0 <= p['acc_by_K']['2'] <= 1 and len(p['jacobian_by_pass']) == 2
  assert math.isfinite(p['jacobian_by_pass'][1]['spectral_norm_median'])
  assert 'loop_probe' not in run_facts(_cfg(ffn='geglu', steps=4, eval_every=2))
