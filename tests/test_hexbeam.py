import pytest
import antenna_designer as ant
from antenna_designer.designs.hexbeam import HexbeamBuilder, get_hexbeam_data_opt

import math

@pytest.mark.skip(reason="Draws to screen")
def test_hexbeam_draw():
    builder = HexbeamBuilder()
    builder.draw(builder.build_wires())

def test_hexbeam_sweep_freq():
  ant.sweep_freq(HexbeamBuilder(), z0=50, rng=(29,31), fn='hexbeam_sweep_freq.pdf')

def test_hexbeam_sweep_freq2():
  ant.sweep(HexbeamBuilder(), 'freq', rng=(29,31), fn='hexbeam_sweep_freq2.pdf')

def test_hexbeam_sweep_halfdriver():
  ant.sweep(HexbeamBuilder(), 'halfdriver', rng=(2.6,3.0), fn='hexbeam_sweep_halfdriver.pdf')

def test_hexbeam_sweep_base():
  ant.sweep(HexbeamBuilder(), 'base', rng=(2.6,3.0), fn='hexbeam_sweep_base.pdf')

def test_hexbeam_sweep_t0_factor():
  ant.sweep(HexbeamBuilder(), 't0_factor', rng=(.05,.25), fn='hexbeam_sweep_t0_factor.pdf')

def test_hexbeam_sweep_tipspacer_factor():
  ant.sweep(HexbeamBuilder(), 'tipspacer_factor', rng=(.05,.25), fn='hexbeam_sweep_tipspacer_factor.pdf')

def test_hexbeam_sweep_gain_freq():
  ant.sweep_gain(HexbeamBuilder(), 'freq', (28,32), fn='hexbeam_sweep_gain_freq.pdf')

def test_hexbeam_sweep_gain_halfdriver():
  ant.sweep_gain(HexbeamBuilder(), 'halfdriver', rng=(2.6,3.0), fn='hexbeam_sweep_gain_halfdriver.pdf')

def test_hexbeam_sweep_gain_base():
  ant.sweep_gain(HexbeamBuilder(), 'base', rng=(2.6,3.0), fn='hexbeam_sweep_gain_base.pdf')

def test_hexbeam_sweep_gain_t0_factor():
  ant.sweep_gain(HexbeamBuilder(), 't0_factor', rng=(.05,.25), fn='hexbeam_sweep_gain_t0_factor.pdf')

def test_hexbeam_sweep_gain_tipspacer_factor():
  ant.sweep_gain(HexbeamBuilder(), 'tipspacer_factor', rng=(.05,.25), fn='hexbeam_sweep_gain_tipspacer_factor.pdf')

def test_hexbeam_pattern():
  ant.pattern(HexbeamBuilder(), fn='hexbeam_pattern.pdf')

def test_hexbeam_pattern3d():
  ant.pattern3d(HexbeamBuilder(), fn='hexbeam_pattern3d.pdf')

def test_hexbeam_optimize():
  params = ant.optimize(HexbeamBuilder(), ['halfdriver','t0_factor','tipspacer_factor'], z0=50, resonance=True, opt_gain=True)

  print(params)

  for k, v in get_hexbeam_data_opt().items():
    assert math.fabs(params[k]-v) < 0.01
