import antenna_designer as ant
from antenna_designer.designs.dipole import Builder

def test_fandipole_sweep_freq():
  ant.sweep_freq(Builder(), z0=50, rng=(10,30), npoints=2, fn='/dev/null')
