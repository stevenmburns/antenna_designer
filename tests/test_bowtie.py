import antenna_designer as ant
from antenna_designer.designs.bowtie import BowtieBuilder

import math

def test_bowtie_sweep_freq():
  ant.sweep_freq(BowtieBuilder(), fn='bowtie_sweep_freq.pdf')

def test_bowtie_pattern():
  ant.pattern(BowtieBuilder(), fn='bowtie_pattern.pdf')

def test_bowtie_pattern3d():
  ant.pattern3d(BowtieBuilder(), fn='bowtie_pattern3d.pdf')

def test_bowtie_sweep_slope():
  ant.sweep(BowtieBuilder(), 'slope', (.2,1), fn='bowtie_sweep_slope.pdf')

def test_bowtie_sweep_length():
  ant.sweep(BowtieBuilder(), 'length', (4,6), fn='sigle_sweep_length.pdf')


def test_bowtie_optimize():
  gold_params = BowtieBuilder.default_params

  params = ant.optimize(BowtieBuilder(gold_params), ['length', 'slope'], z0=200)

  assert all(math.fabs(params[k]-v) < 0.01 for k, v in gold_params.items())


