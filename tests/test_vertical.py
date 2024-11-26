import antenna_designer as ant
from antenna_designer.designs.vertical import Builder

def test_vertical_sweep_freq():
  ant.sweep_freq(Builder(), z0=50, fn='vertical_sweep_freq.pdf')

def test_vertical_sweep_length():
  ant.sweep(Builder(), 'length', rng=(2,3), fn='vertical_sweep_length.pdf')

def test_vertical_pattern():
  ant.pattern(Builder(), fn='vertical_pattern.pdf')

def test_vertical_pattern3d():
  ant.pattern3d(Builder(), fn='vertical_pattern3d.pdf')

def test_vertical_optimize():
  params = ant.optimize(Builder(), ['length'], z0=50, resonance=True)
  assert all(abs(params[k]-v) < 0.01 for k, v in Builder.default_params.items())
