import pytest
import math

import antenna as ant
from bowtie import BowtieArrayBuilder, BowtieArray2x4Builder
from invvee import InvveeArrayBuilder

@pytest.mark.skip(reason="Draws to screen")
def test_bowtiearraybuilder():
    b = BowtieArrayBuilder()
    b.build_wires()

def test_bowtiearray2x4_pattern():
  ant.pattern(BowtieArray2x4Builder(), fn='bowtiearray2x4_pattern.pdf')

@pytest.mark.skip(reason='Takes too long and unstable params')
def test_bowtiearray2x4_optimize():
  gold_params = BowtieArray2x4Builder.default_params

  params = ant.optimize(BowtieArray2x4Builder(gold_params), ['length_otop', 'slope_otop', 'length_obot', 'slope_obot', 'length_itop', 'slope_itop', 'length_ibot', 'slope_ibot'], z0=200, resonance=True, opt_gain=True)

  print(params)

  assert all(math.fabs(params[k]-v) < 0.01 for k, v in gold_params.items())


def test_invveearray_pattern():
  ant.pattern(InvveeArrayBuilder(), fn="invveearray_pattern.pdf")

@pytest.mark.skip(reason='Takes too long and unstable params')
def test_invveearray_optimize():
  gold_params = InvveeArrayBuilder.default_params

  b = InvveeArrayBuilder(gold_params)
  b.draw(b.build_wires())

  params = ant.optimize(b, ['length_top', 'slope_top', 'length_bot', 'slope_bot'], z0=50, resonance=True, opt_gain=True)

  print(params)

  assert all(math.fabs(params[k]-v) < 0.01 for k, v in gold_params.items())

