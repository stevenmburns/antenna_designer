import antenna as ant
from bowtie import BowtieBuilder, get_bowtie_data, BowtieSingleBuilder, get_single_bowtie_data

import math

def test_bowtie_pattern():
  ant.pattern(BowtieBuilder(**get_bowtie_data()), fn='pattern.pdf')

def test_bowtie_pattern3d():
  ant.pattern3d(BowtieBuilder(**get_bowtie_data()), fn='pattern3d.pdf')
  
def test_bowtie_sweep_freq():
  ant.sweep_freq(BowtieBuilder(**get_bowtie_data()), fn='sweep_freq.pdf')

def test_bowtie_sweep_freq2():
  ant.sweep(BowtieBuilder(**get_bowtie_data()), 'freq', (28,29), fn='sweep_freq.pdf')

def test_bowtie_sweep_slope_top():
  ant.sweep(BowtieBuilder(**get_bowtie_data()), 'slope_top', (.2,1), fn='bowtie_sweep_slope_top.pdf')

def test_bowtie_sweep_slope_bot():
  ant.sweep(BowtieBuilder(**get_bowtie_data()), 'slope_bot', (.2,1), fn='bowtie_sweep_slope_bot.pdf')

def test_bowtie_sweep_length_top():
  ant.sweep(BowtieBuilder(**get_bowtie_data()), 'length_top', (4,6), fn='bowtie_sweep_length_top.pdf')

def test_bowtie_sweep_length_bot():
  ant.sweep(BowtieBuilder(**get_bowtie_data()), 'length_bot', (4,6), fn='bowtie_sweep_length_bot.pdf')

def test_bowtie_optimize():
  gold_params = get_bowtie_data()

  params = ant.optimize(BowtieBuilder(**gold_params), ['length_top', 'slope_top', 'length_bot', 'slope_bot'], z0=200)

  for k, v in gold_params.items():
    assert math.fabs(params[k]-v) < 0.01


def test_single_sweep_freq():
  ant.sweep_freq(BowtieSingleBuilder(**get_single_bowtie_data()), fn='single_sweep_freq.pdf')

def test_single_pattern():
  ant.pattern(BowtieSingleBuilder(**get_single_bowtie_data()), fn='single_pattern.pdf')

def test_single_pattern3d():
  ant.pattern3d(BowtieSingleBuilder(**get_single_bowtie_data()), fn='single_pattern3d.pdf')

def test_single_sweep_slope():
  ant.sweep(BowtieSingleBuilder(**get_single_bowtie_data()), 'slope', (.2,1), fn='single_sweep_slope.pdf')

def test_single_sweep_length():
  ant.sweep(BowtieSingleBuilder(**get_single_bowtie_data()), 'length', (4,6), fn='sigle_sweep_length.pdf')


def test_single_optimize():
  gold_params = get_single_bowtie_data()

  params = ant.optimize(BowtieSingleBuilder(**gold_params), ['length', 'slope'], z0=200)

  for k, v in gold_params.items():
    assert math.fabs(params[k]-v) < 0.01


