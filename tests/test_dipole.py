import antenna_designer as ant
from antenna_designer.designs.dipole import Builder

import math

def test_dipole_sweep_freq():
  ant.sweep_freq(Builder(), fn='dipole_sweep_freq.pdf')

def test_dipole_sweep_length():
  ant.sweep(Builder(), 'length', (4,6), fn='dipole_sweep_length.pdf')

def test_dipole_pattern():
  ant.pattern(Builder(), fn='dipole_pattern.pdf')

def test_dipole_pattern3d():
  ant.pattern3d(Builder(), fn='dipole_pattern3d.pdf')

def test_dipole_optimize():
  params = ant.optimize(Builder(), ['length'], z0=50, resonance=True)

  for k, v in Builder.default_params.items():
    assert math.fabs(params[k]-v) < 0.01
