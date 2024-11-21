import antenna as ant
from dipole import DipoleBuilder, get_dipole_data

import math

def test_dipole_sweep_freq():
  ant.sweep_freq(DipoleBuilder(**get_dipole_data()), fn='dipole_sweep_freq.pdf')

def test_dipole_sweep_length():
  ant.sweep(DipoleBuilder(**get_dipole_data()), 'length', (4,6), fn='dipole_sweep_length.pdf')

def test_dipole_pattern():
  ant.pattern(DipoleBuilder(**get_dipole_data()), fn='dipole_pattern.pdf')

def test_dipole_pattern3d():
  ant.pattern3d(DipoleBuilder(**get_dipole_data()), fn='dipole_pattern3d.pdf')

def test_dipole_optimize():
  params = ant.optimize(DipoleBuilder(**get_dipole_data()), ['length'], z0=50, resonance=True)

  for k, v in get_dipole_data().items():
    assert math.fabs(params[k]-v) < 0.01
