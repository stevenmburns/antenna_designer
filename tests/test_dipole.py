import antenna_designer as ant
from antenna_designer.designs.dipole import DipoleBuilder

import math

def test_dipole_sweep_freq():
  ant.sweep_freq(DipoleBuilder(), fn='dipole_sweep_freq.pdf')

def test_dipole_sweep_length():
  ant.sweep(DipoleBuilder(), 'length', (4,6), fn='dipole_sweep_length.pdf')

def test_dipole_pattern():
  ant.pattern(DipoleBuilder(), fn='dipole_pattern.pdf')

def test_dipole_pattern3d():
  ant.pattern3d(DipoleBuilder(), fn='dipole_pattern3d.pdf')

def test_dipole_optimize():
  params = ant.optimize(DipoleBuilder(), ['length'], z0=50, resonance=True)

  for k, v in DipoleBuilder.default_params.items():
    assert math.fabs(params[k]-v) < 0.01
