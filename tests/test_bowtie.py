import antenna_designer as ant
from antenna_designer.designs.bowtie import Builder

import math

def test_bowtie_sweep_freq():
  ant.sweep_freq(Builder(), fn='bowtie_sweep_freq.pdf')

def test_bowtie_pattern():
  ant.pattern(Builder(), fn='bowtie_pattern.pdf')

def test_bowtie_pattern3d():
  ant.pattern3d(Builder(), fn='bowtie_pattern3d.pdf')

def test_bowtie_sweep_slope():
  ant.sweep(Builder(), 'slope', (.2,1), fn='bowtie_sweep_slope.pdf')

def test_bowtie_sweep_length():
  ant.sweep(Builder(), 'length', (4,6), fn='sigle_sweep_length.pdf')


def test_bowtie_optimize():
  gold_params = Builder.default_params

  params = ant.optimize(Builder(gold_params), ['length', 'slope'], z0=200)

  assert all(math.fabs(params[k]-v) < 0.01 for k, v in gold_params.items())


