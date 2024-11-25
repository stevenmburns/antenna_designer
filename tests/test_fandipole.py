import pytest
import antenna_designer as ant
from antenna_designer.designs.fandipole import FandipoleBuilder
from icecream import ic

import math

@pytest.mark.skip(reason="Draws to screen")
def test_fandipole_build():
  b = FandipoleBuilder()
  b.draw(b.build_wires())

def test_fandipole_sweep_freq():
  params = dict(FandipoleBuilder.default_params)
  #params['length_20'] = 10
  #ant.sweep(FandipoleBuilder(params), 'freq', rng=(14,30), npoints=101, fn='fandipole_sweep_freq.pdf')
  ant.sweep_freq(FandipoleBuilder(params), z0=50, rng=(10,30), npoints=101, fn='fandipole_sweep_freq.pdf')

def test_fandipole_sweep_length_20():
  params = dict(FandipoleBuilder.default_params)
  params['freq'] = 14.3
  ant.sweep(FandipoleBuilder(params), 'length_20', (10.5,12), fn='fandipole_sweep_length_20.pdf')

def test_fandipole_sweep_length17():
  params = dict(FandipoleBuilder.default_params)
  params['freq'] = 18.1575
  ant.sweep(FandipoleBuilder(params), 'length_17', (11.3*17/20-1,11.3*17/20+1), fn='fandipole_sweep_length_17.pdf')

def test_fandipole_sweep_length_15():
  params = dict(FandipoleBuilder.default_params)
  params['freq'] = 21.383
  ant.sweep(FandipoleBuilder(params), 'length_15', (11.3*15/20-1,11.3*15/20+1), fn='fandipole_sweep_length_15.pdf')

def test_fandipole_sweep_length_12():
  params = dict(FandipoleBuilder.default_params)

  params = {'base': 7,
             'freq': 28.57,
             'length_10': 3.550404757728437,
             'length_12': 4.165365028613822,
             'length_15': 6.687555534770612,
             'length_17': 7.875750399295038,
             'length_20': 10,
             'slope': 0.604}


  params['freq'] = 24.97
  ant.sweep(FandipoleBuilder(params), 'length_12', (3,6), fn='fandipole_sweep_length_12.pdf')

def test_fandipole_sweep_length_10():
  params = dict(FandipoleBuilder.default_params)

  params = {'base': 7,
             'freq': 28.57,
             'length_10': 3.550404757728437,
             'length_12': 4.165365028613822,
             'length_15': 6.687555534770612,
             'length_17': 7.875750399295038,
             'length_20': 10,
             'slope': 0.604}

  params['freq'] = 28.57
  ant.sweep(FandipoleBuilder(params), 'length_10', (3,6), fn='fandipole_sweep_length_10.pdf')

def test_fandipole_pattern():
  ant.pattern(FandipoleBuilder(), fn='fandipole_pattern.pdf')

def test_fandipole_pattern3d():
  ant.pattern3d(FandipoleBuilder(), fn='fandipole_pattern3d.pdf')

@pytest.mark.skip(reason="Takes too long has unstable params")
def test_fandipole_optimize():
  params = ant.optimize(FandipoleBuilder(), ['length_10', 'length_12'], z0=50, resonance=True)
  ic(params)

  for k, v in FandipoleBuilder.default_params.items():
    assert math.fabs(params[k]-v) < 0.01

@pytest.mark.skip(reason="Takes too long and has unstable params")
def test_fandipole_seq_optimize():

  #params = dict(FandipoleBuilder.default_params)

  gold_params = {'base': 7,
             'freq': 28.57,
             'length_10': 3.550404757728437,
             'length_12': 4.165365028613822,
             'length_15': 6.687555534770612,
             'length_17': 7.875750399295038,
             'length_20': 10,
             'slope': 0.604}


  params = dict(gold_params)

  for iter in range(10):
    for (freq, nm) in (#(14.3, 'length_20'), (18.1575, 'length_17'), (21.383, 'length_15'),
                       (24.97, 'length_12'),
                       (28.57, 'length_10')):
      params['freq'] = freq
      params = ant.optimize(FandipoleBuilder(params), [nm], z0=50, resonance=False)
      ic('results for', freq, nm, 'are', params)

  for k, v in gold_params.items():
    assert math.fabs(params[k]-v) < 0.01
