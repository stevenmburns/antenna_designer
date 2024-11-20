from antenna import *
from dipole import *

def test_dipole_sweep_freq():
  sweep_freq(DipoleBuilder(**get_dipole_data()), fn='dipole_sweep_freq.pdf')

def test_dipole_sweep_length():
  sweep(DipoleBuilder(**get_dipole_data()), 'length', (4,6), fn='dipole_sweep_length.pdf')

def test_dipole_pattern():
  pattern(DipoleBuilder(**get_dipole_data()), fn='dipole_pattern.pdf')

def test_dipole_pattern3d():
  pattern3d(DipoleBuilder(**get_dipole_data()), fn='dipole_pattern3d.pdf')

def test_dipole_optimize():
  params = optimize(DipoleBuilder(**get_dipole_data()), ['length'], z0=50, resonance=True)

  for k, v in get_dipole_data().items():
    assert math.fabs(params[k]-v) < 0.01
