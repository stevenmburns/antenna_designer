import antenna as ant
from vertical import VerticalBuilder, get_vertical_data

import math

def test_vertical_sweep_freq():
  ant.sweep_freq(VerticalBuilder(**get_vertical_data()), z0=50, fn='vertical_sweep_freq.pdf')

def test_vertical_sweep_length():
  ant.sweep(VerticalBuilder(**get_vertical_data()), 'length', (2,3), fn='vertical_sweep_length.pdf')

def test_vertical_pattern():
  ant.pattern(VerticalBuilder(**get_vertical_data()), fn='vertical_pattern.pdf')

def test_vertical_pattern3d():
  ant.pattern3d(VerticalBuilder(**get_vertical_data()), fn='vertical_pattern3d.pdf')

def test_vertical_optimize():
  #bt = ant.Antenna(VerticalBuilder(**get_vertical_data()))
  #bt.draw()
  #del bt

  params = ant.optimize(VerticalBuilder(**get_vertical_data()), ['length'], z0=50, resonance=True)

  for k, v in get_vertical_data().items():
    assert math.fabs(params[k]-v) < 0.01

