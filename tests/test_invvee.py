import antenna as ant
from invvee import InvVeeBuilder

import math

def test_invvee_sweep_freq():
  ant.sweep_freq(InvVeeBuilder(), fn='invvee_sweep_freq.pdf')

def test_invvee_sweep_length():
  ant.sweep(InvVeeBuilder(), 'length', (4,6), fn='invvee_sweep_length.pdf')

def test_invvee_sweep_slope():
  ant.sweep(InvVeeBuilder(), 'slope', (.2,1), fn='invvee_sweep_slope.pdf')

def test_invvee_sweep_gain_freq():
  ant.sweep_gain(InvVeeBuilder(), 'freq', (28,30), fn='invvee_sweep_gain_freq.pdf')

def test_invvee_sweep_gain_length():
  ant.sweep_gain(InvVeeBuilder(), 'length', (4,6), fn='invvee_sweep_gain_length.pdf')

def test_invvee_sweep_gain_slope():
  ant.sweep_gain(InvVeeBuilder(), 'slope', (.2,1), fn='invvee_sweep_gain_slope.pdf')


def test_invvee_optimize():

  gold_params = InvVeeBuilder.default_params

  params = ant.optimize(InvVeeBuilder(gold_params), ['length','slope'], z0=50)

  for k, v in gold_params.items():
    assert math.fabs(params[k]-v) < 0.01
