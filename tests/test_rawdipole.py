import antenna_designer as ant
from icecream import ic
from antenna_designer.designs.rawdipole import Builder

def test_rawdipole_sweep_freq():
  ant.sweep_freq(Builder(), fn='rawdipole_sweep_freq.pdf')

def test_rawdipole_sweep_length():
  ant.sweep(Builder(), 'length', rng=(4,6), fn='rawdipole_sweep_length.pdf')

def test_rawdipole_pattern():
  ant.pattern(Builder(), fn='rawdipole_pattern.pdf')

def test_rawdipole_pattern3d():
  ant.pattern3d(Builder(), fn='rawdipole_pattern3d.pdf')

def test_rawdipole_optimize():
  builder = ant.optimize(Builder(), ['length'], z0=50, resonance=True)
  ic(builder.length, builder.base, builder.freq)

  opt_params = {
    'length': 5.03271484375,
    'base': 7,
    'freq': 28.57
    }

  assert all(abs(getattr(builder,k)-v) < 0.01 for k, v in opt_params.items())
