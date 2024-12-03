import pytest

from antenna_designer import pysim

def test_unit():
    pass

def test_distance():

    delta = 2*pysim.halfdriver/pysim.nsegs

    assert abs(pysim.distance((0,0), (1,0)) - 1*delta) < 0.01
    assert abs(pysim.distance((0,-1), (1,0)) - 1.5*delta) < 0.01

    assert abs(pysim.distance((0,-1), (pysim.nsegs-1,1)) - pysim.nsegs*delta) < 0.01

    with pytest.raises(AssertionError):
        pysim.distance((-1,-1), (0,1))

    midseg_index = pysim.nsegs//2

    assert abs(pysim.delta_l(midseg_index) - delta) < 0.01
    assert abs(pysim.delta_l(midseg_index, adj=-1) - delta) < 0.01
    assert abs(pysim.delta_l(midseg_index, adj=1) - delta) < 0.01
