import antennaknobs as ant
from antennaknobs.designs.dipoles.invvee import Builder

from conftest import needs_pynec


@needs_pynec
def test_fandipole_sweep_freq():
    ant.sweep_freq(Builder(), z0=50, rng=(10, 30), npoints=2, fn="/dev/null")
