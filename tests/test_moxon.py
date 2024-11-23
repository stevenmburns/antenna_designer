import pytest
import antenna as ant
from moxon import MoxonBuilder

import math

@pytest.mark.skip(reason="Displays to screen")
def test_moxon_draw():
    builder = MoxonBuilder()
    builder.draw(builder.build_wires())

def test_moxon_sweep_freq():
  ant.sweep_freq(MoxonBuilder(), z0=50, rng=(28,30), fn='moxon_sweep_freq.pdf')

def test_moxon_sweep_freq2():
  ant.sweep(MoxonBuilder(), 'freq', rng=(28,30), fn='moxon_sweep_freq2.pdf')

def test_moxon_sweep_halfdriver():
  ant.sweep(MoxonBuilder(), 'halfdriver', rng=(2.3,2.6), fn='moxon_sweep_halfdriver.pdf')

def test_moxon_sweep_base():
  ant.sweep(MoxonBuilder(), 'base', rng=(5,10), fn='moxon_sweep_base.pdf')

def test_moxon_sweep_t0_factor():
  ant.sweep(MoxonBuilder(), 't0_factor', rng=(.3,.45), fn='moxon_sweep_t0_factor.pdf')

def test_moxon_sweep_tipspacer_factor():
  ant.sweep(MoxonBuilder(), 'tipspacer_factor', rng=(.05,.15), fn='moxon_sweep_tipspacer_factor.pdf')

def test_moxon_sweep_aspect_ratio():
  ant.sweep(MoxonBuilder(), 'aspect_ratio', rng=(.1,.5), fn='moxon_sweep_aspect_ratio.pdf')

def test_moxon_sweep_gain_freq():
  ant.sweep_gain(MoxonBuilder(), 'freq', (28,30), fn='moxon_sweep_gain_freq.pdf')

def test_moxon_sweep_gain_halfdriver():
  ant.sweep_gain(MoxonBuilder(), 'halfdriver', rng=(2.3,2.6), fn='moxon_sweep_gain_halfdriver.pdf')

def test_moxon_sweep_gain_base():
  ant.sweep_gain(MoxonBuilder(), 'base', rng=(5,10), fn='moxon_sweep_gain_base.pdf')

def test_moxon_sweep_gain_t0_factor():
  ant.sweep_gain(MoxonBuilder(), 't0_factor', rng=(.3,.45), fn='moxon_sweep_gain_t0_factor.pdf')

def test_moxon_sweep_gain_tipspacer_factor():
  ant.sweep_gain(MoxonBuilder(), 'tipspacer_factor', rng=(.05,.15), fn='moxon_sweep_gain_tipspacer_factor.pdf')

def test_moxon_sweep_gain_aspect_ratio():
  ant.sweep_gain(MoxonBuilder(), 'aspect_ratio', rng=(.1,.5), fn='moxon_sweep_gain_aspect_ratio.pdf')

def test_moxon_pattern():
  ant.pattern(MoxonBuilder(), fn='moxon_pattern.pdf')

def test_moxon_pattern3d():
  ant.pattern3d(MoxonBuilder(), fn='moxon_pattern3d.pdf')

@pytest.mark.skip(reason="Too long and golden answer changing")
def test_moxon_optimize():
  params = ant.optimize(MoxonBuilder(), ['halfdriver','t0_factor','tipspacer_factor'], z0=50, opt_gain=True)

  print(params)

  for k, v in MoxonBuilder.opt_params.items():
    assert math.fabs(params[k]-v) < 0.01
