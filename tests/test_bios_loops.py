"""rcdl.bios looping: K = 1 is the plain model, K passes reuse the last FFN, the probe's Jacobians and K sweep."""

import math

import pytest
import torch

from rcdl.bios import BiosConfig, BioWorld, GemmaBios, evaluate, loop_probe, run_bios

BASE = dict(n_people=32, seq_len=128, batch_size=8, d_model=32, layers=2, heads=2, kv_heads=1, head_dim=16,
            ffn_width=64, steps=20, eval_every=10, eval_bios=16, lr=3e-3, cos_width=128)


def _model(**kw):
  cfg = BiosConfig(**(BASE | kw))
  world = BioWorld(cfg.n_people, cfg.world_seed, 'cpu')
  m = GemmaBios(cfg, world.vocab_size)
  with torch.no_grad():                               # move the norm scales away from 0 so the passes differ
    for n, p in m.named_parameters():
      if p.dim() == 1:
        p.add_(0.3 * torch.randn(p.shape, generator=torch.Generator().manual_seed(len(n))))
  return cfg, world, m.eval()


def _reference(m, tokens):
  """The forward pass before looping existed (one pass of every FFN)."""
  x = m.embed(tokens) * math.sqrt(m.cfg.d_model)
  for blk in m.blocks:
    cos, sin = m.rope[1] if blk.is_global else m.rope[0]
    x = x + blk.post_attn_norm(blk.attn(blk.input_norm(x), cos, sin, blk.window))
    x = x + blk.post_ffn_norm(blk.ffn(blk.pre_ffn_norm(x)))
  return m.final_norm(x) @ m.embed.weight.T


def _tokens(world, n=4):
  return world.eval_rows(torch.arange(n), 'eval', torch.Generator().manual_seed(5))


@pytest.mark.parametrize('ffn', ['geglu', 'cos'])
def test_one_pass_is_the_plain_model_and_k_passes_reuse_the_last_ffn(ffn):
  cfg, world, m = _model(ffn=ffn)
  x, _, _, _, route = _tokens(world)
  with torch.no_grad():
    torch.testing.assert_close(m(x, route), _reference(m, x), rtol=0, atol=0)
    a = 0.5
    _, _, m3 = _model(ffn=ffn, ffn_loops=3, ffn_loop_alpha=a)
    m3.load_state_dict(m.state_dict())
    assert [b.loops for b in m3.blocks] == [1, 3]
    h = m3.loop_input(x, route)
    last = m3.blocks[-1]
    for _ in range(3):
      h = h + a * last.post_ffn_norm(last.ffn(last.pre_ffn_norm(h)))
    torch.testing.assert_close(m3(x, route), m3.final_norm(h) @ m3.embed.weight.T)
    last.loops = 1
    assert not torch.allclose(m3(x, route), m3.final_norm(h) @ m3.embed.weight.T)


def test_looping_needs_a_geglu_or_cos_last_layer():
  for kw in (dict(special='rbf', rbf_capacity=1000), dict(special='geglu_fact')):
    with pytest.raises(ValueError):
      _model(ffn_loops=2, **kw)
  _model(ffn_loops=1, special='geglu_fact')            # K = 1 is allowed everywhere


@pytest.mark.parametrize('ffn', ['geglu', 'cos'])
def test_probe_matches_finite_differences_and_restores_k(ffn):
  cfg, world, m = _model(ffn=ffn, ffn_loops=2, ffn_loop_alpha=0.7)
  m = m.double()
  data = _tokens(world, 8)
  out = loop_probe(m, data, data, n_jac=5, seed=3)
  assert m.blocks[-1].loops == 2 and set(out['acc_by_K']) == {'1', '2', '4', '8'}
  assert out['acc_by_K']['2']['attr_acc'] == evaluate(m, data)['attr_acc'] and 'seen_attr_acc' in out['acc_by_K']['8']
  inputs, _, scored, _, route = data
  blk = m.blocks[-1]
  step = lambda v: v + 0.7 * blk.post_ffn_norm(blk.ffn(blk.pre_ffn_norm(v)))
  with torch.no_grad():
    x = m.loop_input(inputs, route)[scored]
  h = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(3))[:5]]
  rads, norms, fd_norms = [], [], []
  for v in h:                                        # GemmaRMSNorm computes in fp32, so differences need a larger step
    J = torch.autograd.functional.jacobian(step, v)
    fd = torch.stack([(step(v + 1e-3 * e) - step(v - 1e-3 * e)) / 2e-3 for e in torch.eye(len(v), dtype=v.dtype)], 1)
    rads.append(torch.linalg.eigvals(J).abs().max())
    norms.append(torch.linalg.matrix_norm(J, ord=2))
    fd_norms.append(torch.linalg.matrix_norm(fd.detach(), ord=2))
  p1 = out['jacobian_by_pass'][0]
  assert p1['spectral_radius_median'] == pytest.approx(float(torch.stack(rads).median()), rel=1e-6)
  assert p1['spectral_norm_median'] == pytest.approx(float(torch.stack(norms).median()), rel=1e-6)
  assert p1['spectral_norm_median'] == pytest.approx(float(torch.stack(fd_norms).median()), rel=2e-3)
  assert len(out['jacobian_by_pass']) == 2 and out['n_jac'] == 5


def test_tiny_run_reports_the_probe_only_when_asked():
  r = run_bios(BiosConfig(**(BASE | dict(ffn='cos', ffn_loops=2, final_probe=True, eval_seen_templates=True))))
  p = r['loop_probe']
  assert 0 <= p['acc_by_K']['2']['attr_acc'] <= 1 and len(p['jacobian_by_pass']) == 2
  assert p['acc_by_K']['2']['attr_acc'] == r['final_attr_acc']
  assert math.isfinite(p['jacobian_by_pass'][1]['spectral_norm_median'])
  assert 'loop_probe' not in run_bios(BiosConfig(**(BASE | dict(steps=4, eval_every=2))))
