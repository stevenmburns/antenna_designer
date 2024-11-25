import antenna_designer as ant
from antenna_designer.designs.invvee import Builder

import math

def test_invvee_sweep_freq():
  ant.sweep_freq(Builder(), fn='invvee_sweep_freq.pdf')

def test_invvee_sweep_length():
  ant.sweep(Builder(), 'length', (4,6), fn='invvee_sweep_length.pdf')

def test_invvee_sweep_slope():
  ant.sweep(Builder(), 'slope', (.2,1), fn='invvee_sweep_slope.pdf')

def test_invvee_sweep_gain_freq():
  ant.sweep_gain(Builder(), 'freq', (28,30), fn='invvee_sweep_gain_freq.pdf')

def test_invvee_sweep_gain_length():
  ant.sweep_gain(Builder(), 'length', (4,6), fn='invvee_sweep_gain_length.pdf')

def test_invvee_sweep_gain_slope():
  ant.sweep_gain(Builder(), 'slope', (.2,1), fn='invvee_sweep_gain_slope.pdf')


def test_invvee_optimize():

  gold_params = Builder.default_params

  params = ant.optimize(Builder(gold_params), ['length','slope'], z0=50)

  for k, v in gold_params.items():
    assert math.fabs(params[k]-v) < 0.01
