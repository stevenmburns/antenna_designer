import pytest
import antenna_designer as ant
from antenna_designer.designs.fandipole import Builder
from icecream import ic

@pytest.mark.skip(reason="Draws to screen")
def test_fandipole_build():
  b = Builder()
  b.draw(b.build_wires())

def test_fandipole_sweep_freq():
  params = dict(Builder.default_params)
  #params['length_20'] = 10
  #ant.sweep(Builder(params), 'freq', rng=(14,30), npoints=101, fn='fandipole_sweep_freq.pdf')
  ant.sweep_freq(Builder(params), z0=50, rng=(10,30), npoints=101, fn='fandipole_sweep_freq.pdf')

def test_fandipole_sweep_length_20():
  params = dict(Builder.default_params)
  params['freq'] = 14.3
  ant.sweep(Builder(params), 'length_20', rng=(10.5,12), fn='fandipole_sweep_length_20.pdf')

def test_fandipole_sweep_length17():
  params = dict(Builder.default_params)
  params['freq'] = 18.1575
  ant.sweep(Builder(params), 'length_17', rng=(11.3*17/20-1,11.3*17/20+1), fn='fandipole_sweep_length_17.pdf')

def test_fandipole_sweep_length_15():
  params = dict(Builder.default_params)
  params['freq'] = 21.383
  ant.sweep(Builder(params), 'length_15', rng=(11.3*15/20-1,11.3*15/20+1), fn='fandipole_sweep_length_15.pdf')

def test_fandipole_sweep_length_12():
  params = dict(Builder.default_params)

  params = {'base': 7,
             'freq': 28.57,
             'length_10': 3.550404757728437,
             'length_12': 4.165365028613822,
             'length_15': 6.687555534770612,
             'length_17': 7.875750399295038,
             'length_20': 10,
             'slope': 0.604}


  params['freq'] = 24.97
  ant.sweep(Builder(params), 'length_12', rng=(3,6), fn='fandipole_sweep_length_12.pdf')

def test_fandipole_sweep_length_10():
  params = dict(Builder.default_params)

  params = {'base': 7,
             'freq': 28.57,
             'length_10': 3.550404757728437,
             'length_12': 4.165365028613822,
             'length_15': 6.687555534770612,
             'length_17': 7.875750399295038,
             'length_20': 10,
             'slope': 0.604}

  params['freq'] = 28.57
  ant.sweep(Builder(params), 'length_10', rng=(3,6), fn='fandipole_sweep_length_10.pdf')

def test_fandipole_pattern():
  ant.pattern(Builder(), fn='fandipole_pattern.pdf')

def test_fandipole_pattern3d():
  ant.pattern3d(Builder(), fn='fandipole_pattern3d.pdf')

@pytest.mark.skip(reason="Takes too long has unstable params")
def test_fandipole_optimize():
  params = ant.optimize(Builder(), ['length_10', 'length_12'], z0=50, resonance=True)
  ic(params)

  assert all(abs(params[k]-v) < 0.01 for k, v in Builder.default_params.items())

@pytest.mark.skip(reason="Takes too long and has unstable params")
def test_fandipole_seq_optimize():

  #params = dict(Builder.default_params)

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
      params = ant.optimize(Builder(params), [nm], z0=50, resonance=False)
      ic('results for', freq, nm, 'are', params)


    assert all(abs(params[k]-v) < 0.01 for k, v in gold_params.items())
