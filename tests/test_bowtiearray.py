import pytest
import antenna_designer as ant
from antenna_designer.designs import bowtiearray

def test_bowtiearray_pattern():
  ant.pattern(bowtiearray.Builder(), fn='bowtiearray_pattern.pdf')

def test_bowtiearray_pattern3d():
  ant.pattern3d(bowtiearray.Builder(), fn='bowtiearray_pattern3d.pdf')
  
def test_bowtiearray_sweep_freq():
  ant.sweep_freq(bowtiearray.Builder(), fn='bowtiearray_sweep_freq.pdf')

def test_bowtiearray_sweep_freq2():
  ant.sweep(bowtiearray.Builder(), 'freq', rng=(28,29), fn='bowtiearray_sweep_freq.pdf')

def test_bowtiearray_sweep_slope_top():
  ant.sweep(bowtiearray.Builder(), 'slope_top', rng=(.2,1), fn='bowtiearray_sweep_slope_top.pdf')

def test_bowtiearray_sweep_slope_bot():
  ant.sweep(bowtiearray.Builder(), 'slope_bot', rng=(.2,1), fn='bowtiearray_sweep_slope_bot.pdf')

def test_bowtiearray_sweep_length_top():
  ant.sweep(bowtiearray.Builder(), 'length_top', rng=(4,6), fn='bowtiearray_sweep_length_top.pdf')

def test_bowtiearray_sweep_length_bot():
  ant.sweep(bowtiearray.Builder(), 'length_bot', rng=(4,6), fn='bowtiearray_sweep_length_bot.pdf')

def test_bowtiearray_sweep_del_z():
  ant.sweep(bowtiearray.Builder(), 'del_z', rng=(1,3), fn='bowtiearray_sweep_del_z.pdf')

def test_bowtiearray_sweep_del_y():
  ant.sweep(bowtiearray.Builder(), 'del_y', rng=(3.5,6), fn='bowtiearray_sweep_del_y.pdf')

def test_bowtiearray_sweep_base():
  ant.sweep(bowtiearray.Builder(), 'base', rng=(6,8), fn='bowtiearray_sweep_base.pdf')

def test_bowtiearray_sweep_gain_del_z():
  ant.sweep_gain(bowtiearray.Builder(), 'del_z', rng=(1,3), fn='bowtiearray_sweep_gain_del_z.pdf')

def test_bowtiearray_sweep_gain_del_y():
  ant.sweep_gain(bowtiearray.Builder(), 'del_y', rng=(3.5,6), fn='bowtiearray_sweep_gain_del_y.pdf')

def test_bowtiearray_sweep_gain_base():
  ant.sweep_gain(bowtiearray.Builder(), 'base', rng=(6,8), fn='bowtiearray_sweep_gain_base.pdf')

def test_bowtiearray_optimize():
  gold_params = bowtiearray.Builder.default_params

  builder = ant.optimize(bowtiearray.Builder(gold_params), ['length_top', 'slope_top', 'length_bot', 'slope_bot'], z0=200)
  params = builder._params
  print(params)

  assert all(abs(params[k]-v) < 0.01 for k, v in gold_params.items())

@pytest.mark.skip(reason="Too long and unstable params")
def test_bowtiearray_optimize_for_gain():
  gold_params = bowtiearray.Builder.default_params

  builder = ant.optimize(bowtiearray.Builder(gold_params), ['length_top', 'slope_top', 'length_bot', 'slope_bot', 'del_y', 'del_z'], z0=200, resonance=True, opt_gain=True)

  params = builder._params
  print(params)

  assert all(abs(params[k]-v) < 0.01 for k, v in gold_params.items())
