import antenna_designer as ant
from antenna_designer.designs.bowtie import Builder

def test_bowtie_sweep_freq():
  ant.sweep_freq(Builder(), fn='bowtie_sweep_freq.pdf')

def test_bowtie_pattern():
  ant.pattern(Builder(), fn='bowtie_pattern.pdf')

def test_bowtie_pattern3d():
  ant.pattern3d(Builder(), fn='bowtie_pattern3d.pdf')

def test_bowtie_sweep_slope():
  ant.sweep(Builder(), 'slope', rng=(.2,1), fn='bowtie_sweep_slope.pdf')

def test_bowtie_sweep_length():
  ant.sweep(Builder(), 'length', rng=(4,6), fn='sigle_sweep_length.pdf')


def test_bowtie_optimize():
  gold_params = Builder.default_params

  builder = ant.optimize(Builder(gold_params), ['length', 'slope'], z0=200)

  params = builder._params

  assert all(abs(params[k]-v) < 0.01 for k, v in gold_params.items())


