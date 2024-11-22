import antenna as ant
from bowtie import BowtieArrayBuilder

import math

def test_bowtiearray_pattern():
  ant.pattern(BowtieArrayBuilder(), fn='bowtiearray_pattern.pdf')

def test_bowtiearray_pattern3d():
  ant.pattern3d(BowtieArrayBuilder(), fn='bowtiearray_pattern3d.pdf')
  
def test_bowtiearray_sweep_freq():
  ant.sweep_freq(BowtieArrayBuilder(), fn='bowtiearray_sweep_freq.pdf')

def test_bowtiearray_sweep_freq2():
  ant.sweep(BowtieArrayBuilder(), 'freq', (28,29), fn='bowtiearray_sweep_freq.pdf')

def test_bowtiearray_sweep_slope_top():
  ant.sweep(BowtieArrayBuilder(), 'slope_top', (.2,1), fn='bowtiearray_sweep_slope_top.pdf')

def test_bowtiearray_sweep_slope_bot():
  ant.sweep(BowtieArrayBuilder(), 'slope_bot', (.2,1), fn='bowtiearray_sweep_slope_bot.pdf')

def test_bowtiearray_sweep_length_top():
  ant.sweep(BowtieArrayBuilder(), 'length_top', (4,6), fn='bowtiearray_sweep_length_top.pdf')

def test_bowtiearray_sweep_length_bot():
  ant.sweep(BowtieArrayBuilder(), 'length_bot', (4,6), fn='bowtiearray_sweep_length_bot.pdf')

def test_bowtiearray_sweep_del_z():
  ant.sweep(BowtieArrayBuilder(), 'del_z', (1,3), fn='bowtiearray_sweep_del_z.pdf')

def test_bowtiearray_sweep_del_y():
  ant.sweep(BowtieArrayBuilder(), 'del_y', (3.5,6), fn='bowtiearray_sweep_del_y.pdf')

def test_bowtiearray_sweep_base():
  ant.sweep(BowtieArrayBuilder(), 'base', (6,8), fn='bowtiearray_sweep_base.pdf')

def test_bowtiearray_sweep_gain_del_z():
  ant.sweep_gain(BowtieArrayBuilder(), 'del_z', (1,3), fn='bowtiearray_sweep_gain_del_z.pdf')

def test_bowtiearray_sweep_gain_del_y():
  ant.sweep_gain(BowtieArrayBuilder(), 'del_y', (3.5,6), fn='bowtiearray_sweep_gain_del_y.pdf')

def test_bowtiearray_sweep_gain_base():
  ant.sweep_gain(BowtieArrayBuilder(), 'base', (6,8), fn='bowtiearray_sweep_gain_base.pdf')

def test_bowtiearray_optimize():
  gold_params = BowtieArrayBuilder.default_params

  params = ant.optimize(BowtieArrayBuilder(gold_params), ['length_top', 'slope_top', 'length_bot', 'slope_bot'], z0=200)

  assert all(math.fabs(params[k]-v) < 0.01 for k, v in gold_params.items())

def test_bowtiearray_optimize_for_gain():
  gold_params = BowtieArrayBuilder.default_params

  params = ant.optimize(BowtieArrayBuilder(gold_params), ['length_top', 'slope_top', 'length_bot', 'slope_bot', 'del_y', 'del_z'], z0=200, resonance=True, opt_gain=True)

  assert all(math.fabs(params[k]-v) < 0.01 for k, v in gold_params.items())
