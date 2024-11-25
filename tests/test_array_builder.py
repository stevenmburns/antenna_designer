import pytest
import math

import antenna_designer as ant

from antenna_designer.designs import bowtiearray, bowtiearray2x4, invveearray

@pytest.mark.skip(reason="Draws t screen")
def test_bowtiearraybuilder():
    b = bowtiearray.Builder()
    b.build_wires()

def test_bowtiearray2x4_pattern():
  ant.pattern(bowtiearray2x4.Builder(), fn='bowtiearray2x4_pattern.pdf')

@pytest.mark.skip(reason='Takes too long and unstable params')
def test_bowtiearray2x4_optimize():
  gold_params = bowtiearray2x4.Builder.default_params

  params = ant.optimize(bowtiearray2x4.Builder(gold_params), ['length_otop', 'slope_otop', 'length_obot', 'slope_obot', 'length_itop', 'slope_itop', 'length_ibot', 'slope_ibot'], z0=200, resonance=True, opt_gain=True)

  print(params)

  assert all(math.fabs(params[k]-v) < 0.01 for k, v in gold_params.items())


def test_invveearray_pattern():
  ant.pattern(invveearray.Builder(), fn="invveearray_pattern.pdf")

@pytest.mark.skip(reason='Takes too long and unstable params')
def test_invveearray_optimize():
  gold_params = invveearray.Builder.default_params

  b = invveearray.Builder(gold_params)
  b.draw(b.build_wires())

  params = ant.optimize(b, ['length_top', 'slope_top', 'length_bot', 'slope_bot'], z0=50, resonance=True, opt_gain=True)

  print(params)

  assert all(math.fabs(params[k]-v) < 0.01 for k, v in gold_params.items())

