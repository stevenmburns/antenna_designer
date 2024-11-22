import antenna as ant
from vertical import VerticalBuilder

import math

def test_vertical_sweep_freq():
  ant.sweep_freq(VerticalBuilder(), z0=50, fn='vertical_sweep_freq.pdf')

def test_vertical_sweep_length():
  ant.sweep(VerticalBuilder(), 'length', (2,3), fn='vertical_sweep_length.pdf')

def test_vertical_pattern():
  ant.pattern(VerticalBuilder(), fn='vertical_pattern.pdf')

def test_vertical_pattern3d():
  ant.pattern3d(VerticalBuilder(), fn='vertical_pattern3d.pdf')

def test_vertical_optimize():
  #bt = ant.Antenna(VerticalBuilder())
  #bt.draw()
  #del bt

  params = ant.optimize(VerticalBuilder(), ['length'], z0=50, resonance=True)

  for k, v in VerticalBuilder.default_params.items():
    assert math.fabs(params[k]-v) < 0.01

