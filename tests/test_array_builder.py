import pytest
import numpy as np
import antenna_designer as ant

from antenna_designer.designs import bowtiearray, bowtiearray2x4, invveearray

@pytest.mark.skip(reason='Takes too long and unstable params')
def test_bowtiearray2x4_optimize():
  gold_params = bowtiearray2x4.Builder.default_params

  builder = ant.optimize(bowtiearray2x4.Builder(gold_params), ['length_otop', 'slope_otop', 'length_obot', 'slope_obot', 'length_itop', 'slope_itop', 'length_ibot', 'slope_ibot'], z0=200, resonance=True, opt_gain=True)

  params = builder._params
  print(params)

  assert all(abs(params[k]-v) < 0.01 for k, v in gold_params.items())

@pytest.mark.skip(reason='Whole file is skipped')
def test_invveearray_pattern():
  ant.pattern(invveearray.Builder(), fn="invveearray_pattern.pdf")

@pytest.mark.skip(reason='Takes too long and unstable params')
def test_invveearray_optimize():
  gold_params = invveearray.Builder.default_params

  b = invveearray.Builder(gold_params)
  b.draw(b.build_wires())

  params = ant.optimize(b, ['length_top', 'slope_top', 'length_bot', 'slope_bot'], z0=50, resonance=True, opt_gain=True)

  print(params)

  assert all(abs(params[k]-v) < 0.01 for k, v in gold_params.items())

@pytest.mark.skip(reason='Draws to screen')
def test_bowtiearray_phase_lr_pattern():

  builders = (
    bowtiearray.Builder(dict(bowtiearray.Builder.default_params, **{'phase_lr': p})) for p in np.linspace(0, 180-36, 5)
  )

  ant.compare_patterns(builders)

@pytest.mark.skip(reason='Draws to screen')
def test_bowtiearray_phase_tb_pattern():

  builders = (
    bowtiearray.Builder(dict(bowtiearray.Builder.default_params, **{'phase_tb': p})) for p in np.linspace(-60, 60, 5)
  )

  ant.compare_patterns(builders)

@pytest.mark.skip(reason="Too long and unstable params")
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
