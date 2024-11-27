import pytest

import antenna_designer as ant
from antenna_designer.designs.invvee import Builder

def test_invvee_sweep_freq():
  ant.sweep_freq(Builder(), fn='invvee_sweep_freq.pdf')

def test_invvee_sweep_length():
  ant.sweep(Builder(), 'length', rng=(4,6), fn='invvee_sweep_length.pdf')

def test_invvee_sweep_slope():
  ant.sweep(Builder(), 'slope', rng=(.2,1), fn='invvee_sweep_slope.pdf')

def test_invvee_sweep_gain_freq():
  ant.sweep_gain(Builder(), 'freq', rng=(28,30), fn='invvee_sweep_gain_freq.pdf')

def test_invvee_sweep_gain_length():
  ant.sweep_gain(Builder(), 'length', rng=(4,6), fn='invvee_sweep_gain_length.pdf')

def test_invvee_sweep_gain_slope():
  ant.sweep_gain(Builder(), 'slope', rng=(.2,1), fn='invvee_sweep_gain_slope.pdf')

def test_invvee_optimize():

  gold_params = Builder.default_params
  builder = ant.optimize(Builder(gold_params), ['length','slope'], z0=50)
  params = builder._params
  print(params)
  assert all(abs(params[k]-v) < 0.01  for k, v in gold_params.items())

@pytest.mark.skip(reason='Experimental')
def test_invvee_diff_optimize():

  gold_params = Builder.default_params
  bounds = ((gold_params['length']*.8, gold_params['length']*1.25),(0,1))

  builders = (
    ant.optimize(Builder(dict(gold_params, **{'base': base})),
                 ['length','slope'], z0=50, bounds=bounds
    ) for base in [5,6,7,8]
  )

  ant.compare_patterns(builders)
