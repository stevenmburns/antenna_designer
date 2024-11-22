import antenna as ant
from bowtie import BowtieBuilder, BowtieSingleBuilder

import math

def test_bowtie_pattern():
  ant.pattern(BowtieBuilder(), fn='pattern.pdf')

def test_bowtie_pattern3d():
  ant.pattern3d(BowtieBuilder(), fn='pattern3d.pdf')
  
def test_bowtie_sweep_freq():
  ant.sweep_freq(BowtieBuilder(), fn='sweep_freq.pdf')

def test_bowtie_sweep_freq2():
  ant.sweep(BowtieBuilder(), 'freq', (28,29), fn='sweep_freq.pdf')

def test_bowtie_sweep_slope_top():
  ant.sweep(BowtieBuilder(), 'slope_top', (.2,1), fn='bowtie_sweep_slope_top.pdf')

def test_bowtie_sweep_slope_bot():
  ant.sweep(BowtieBuilder(), 'slope_bot', (.2,1), fn='bowtie_sweep_slope_bot.pdf')

def test_bowtie_sweep_length_top():
  ant.sweep(BowtieBuilder(), 'length_top', (4,6), fn='bowtie_sweep_length_top.pdf')

def test_bowtie_sweep_length_bot():
  ant.sweep(BowtieBuilder(), 'length_bot', (4,6), fn='bowtie_sweep_length_bot.pdf')

def test_bowtie_sweep_del_z():
  ant.sweep(BowtieBuilder(), 'del_z', (1,3), fn='bowtie_sweep_del_z.pdf')

def test_bowtie_sweep_del_y():
  ant.sweep(BowtieBuilder(), 'del_y', (3.5,6), fn='bowtie_sweep_del_y.pdf')

def test_bowtie_sweep_base():
  ant.sweep(BowtieBuilder(), 'base', (6,8), fn='bowtie_sweep_base.pdf')

def test_bowtie_sweep_gain_del_z():
  ant.sweep_gain(BowtieBuilder(), 'del_z', (1,3), fn='bowtie_sweep_gain_del_z.pdf')

def test_bowtie_sweep_gain_del_y():
  ant.sweep_gain(BowtieBuilder(), 'del_y', (3.5,6), fn='bowtie_sweep_gain_del_y.pdf')

def test_bowtie_sweep_gain_base():
  ant.sweep_gain(BowtieBuilder(), 'base', (6,8), fn='bowtie_sweep_gain_base.pdf')

def test_bowtie_optimize():
  gold_params = BowtieBuilder.default_params

  params = ant.optimize(BowtieBuilder(gold_params), ['length_top', 'slope_top', 'length_bot', 'slope_bot'], z0=200)

  assert all(math.fabs(params[k]-v) < 0.01 for k, v in gold_params.items())


def test_single_sweep_freq():
  ant.sweep_freq(BowtieSingleBuilder(), fn='single_sweep_freq.pdf')

def test_single_pattern():
  ant.pattern(BowtieSingleBuilder(), fn='single_pattern.pdf')

def test_single_pattern3d():
  ant.pattern3d(BowtieSingleBuilder(), fn='single_pattern3d.pdf')

def test_single_sweep_slope():
  ant.sweep(BowtieSingleBuilder(), 'slope', (.2,1), fn='single_sweep_slope.pdf')

def test_single_sweep_length():
  ant.sweep(BowtieSingleBuilder(), 'length', (4,6), fn='sigle_sweep_length.pdf')


def test_single_optimize():
  gold_params = BowtieSingleBuilder.default_params

  params = ant.optimize(BowtieSingleBuilder(gold_params), ['length', 'slope'], z0=200)

  assert all(math.fabs(params[k]-v) < 0.01 for k, v in gold_params.items())


