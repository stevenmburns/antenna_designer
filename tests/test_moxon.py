import pytest
import antenna_designer as ant
from antenna_designer.designs.moxon import Builder

import math

@pytest.mark.skip(reason="Displays to screen")
def test_moxon_draw():
    builder = Builder()
    builder.draw(builder.build_wires())

def test_moxon_sweep_freq():
  ant.sweep_freq(Builder(), z0=50, rng=(28,30), fn='moxon_sweep_freq.pdf')

def test_moxon_sweep_freq2():
  ant.sweep(Builder(), 'freq', rng=(28,30), fn='moxon_sweep_freq2.pdf')

def test_moxon_sweep_halfdriver():
  ant.sweep(Builder(), 'halfdriver', rng=(2.3,2.6), fn='moxon_sweep_halfdriver.pdf')

def test_moxon_sweep_base():
  ant.sweep(Builder(), 'base', rng=(5,10), fn='moxon_sweep_base.pdf')

def test_moxon_sweep_t0_factor():
  ant.sweep(Builder(), 't0_factor', rng=(.3,.45), fn='moxon_sweep_t0_factor.pdf')

def test_moxon_sweep_tipspacer_factor():
  ant.sweep(Builder(), 'tipspacer_factor', rng=(.05,.15), fn='moxon_sweep_tipspacer_factor.pdf')

def test_moxon_sweep_aspect_ratio():
  ant.sweep(Builder(), 'aspect_ratio', rng=(.1,.5), fn='moxon_sweep_aspect_ratio.pdf')

def test_moxon_sweep_gain_freq():
  ant.sweep_gain(Builder(), 'freq', rng=(28,30), fn='moxon_sweep_gain_freq.pdf')

def test_moxon_sweep_gain_halfdriver():
  ant.sweep_gain(Builder(), 'halfdriver', rng=(2.3,2.6), fn='moxon_sweep_gain_halfdriver.pdf')

def test_moxon_sweep_gain_base():
  ant.sweep_gain(Builder(), 'base', rng=(5,10), fn='moxon_sweep_gain_base.pdf')

def test_moxon_sweep_gain_t0_factor():
  ant.sweep_gain(Builder(), 't0_factor', rng=(.3,.45), fn='moxon_sweep_gain_t0_factor.pdf')

def test_moxon_sweep_gain_tipspacer_factor():
  ant.sweep_gain(Builder(), 'tipspacer_factor', rng=(.05,.15), fn='moxon_sweep_gain_tipspacer_factor.pdf')

def test_moxon_sweep_gain_aspect_ratio():
  ant.sweep_gain(Builder(), 'aspect_ratio', rng=(.1,.5), fn='moxon_sweep_gain_aspect_ratio.pdf')

def test_moxon_pattern():
  ant.pattern(Builder(), fn='moxon_pattern.pdf')

def test_moxon_pattern3d():
  ant.pattern3d(Builder(), fn='moxon_pattern3d.pdf')

@pytest.mark.skip(reason="Too long and golden answer changing")
def test_moxon_optimize():
  params = ant.optimize(Builder(), ['halfdriver','t0_factor','tipspacer_factor'], z0=50, opt_gain=True)

  print(params)

  for k, v in Builder.opt_params.items():
    assert math.fabs(params[k]-v) < 0.01
