import pytest

from antenna_designer import pysim

def test_unit():
    pass

def test_distance():

    assert abs(pysim.distance((0,0), (1,0)) - 1) < 0.01
    assert abs(pysim.distance((0,-1), (1,0)) - 1.5) < 0.01

    assert abs(pysim.distance((0,-1), (pysim.nsegs-1,1)) - pysim.nsegs) < 0.01

    with pytest.raises(AssertionError):
        pysim.distance((-1,-1), (10,1))

    midseg_index = pysim.nsegs//2

    assert abs(pysim.delta_l(midseg_index) - 1) < 0.01
    assert abs(pysim.delta_l(midseg_index, adj=-1) - 1) < 0.01
    assert abs(pysim.delta_l(midseg_index, adj=1) - 1) < 0.01
