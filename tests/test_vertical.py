from antenna import *
from vertical import *

def test_vertical_sweep_freq():
  sweep_freq(VerticalBuilder(**get_vertical_data()), z0=50, fn='vertical_sweep_freq.pdf')

def test_vertical_sweep_length():
  sweep(VerticalBuilder(**get_vertical_data()), 'length', (2,3), fn='vertical_sweep_length.pdf')

def test_vertical_pattern():
  pattern(VerticalBuilder(**get_vertical_data()), fn='vertical_pattern.pdf')

def test_vertical_pattern3d():
  pattern3d(VerticalBuilder(**get_vertical_data()), fn='vertical_pattern3d.pdf')

def test_vertical_optimize():
  #bt = Antenna(VerticalBuilder(**get_vertical_data()))
  #bt.draw()
  #del bt

  params = optimize(VerticalBuilder(**get_vertical_data()), ['length'], z0=50, resonance=True)

  for k, v in get_vertical_data().items():
    assert math.fabs(params[k]-v) < 0.01

