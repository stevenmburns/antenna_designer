import antenna_designer as ant
from antenna_designer.designs.dipole import Builder

def test_dipole_sweep_freq():
  ant.sweep_freq(Builder(), fn='dipole_sweep_freq.pdf')

def test_dipole_sweep_length():
  ant.sweep(Builder(), 'length', rng=(4,6), fn='dipole_sweep_length.pdf')

def test_dipole_pattern():
  ant.pattern(Builder(), fn='dipole_pattern.pdf')

def test_dipole_pattern3d():
  ant.pattern3d(Builder(), fn='dipole_pattern3d.pdf')

def test_dipole_optimize():
  builder = ant.optimize(Builder(), ['length'], z0=50, resonance=True)
  assert all(abs(getattr(builder,k)-v) < 0.01 for k, v in Builder.default_params.items())
