#__all__ = []

from .core import Antenna, AntennaBuilder, Array2x2Builder, Array2x4Builder, compare_patterns, sweep, sweep_freq, sweep_gain, optimize, pattern, pattern3d

#from .designs import vertical, moxon, hexbeam, invvee, dipole, bowtie

version = "1.0.0"
print(f"Welcome to antenna_designer version {version}")
