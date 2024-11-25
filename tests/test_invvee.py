import antenna_designer as ant
from antenna_designer.designs.invvee import Builder

import math

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

  params = ant.optimize(Builder(gold_params), ['length','slope'], z0=50)

  assert all(math.fabs(params[k]-v) < 0.01  for k, v in gold_params.items())
