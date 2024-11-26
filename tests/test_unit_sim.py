
from antenna_designer import Antenna
from antenna_designer.designs.dipole import Builder

from unittest.mock import patch

class FakeSC:
    def get_current_segment_tag(self):
        return [1, 1, 2, 3, 3, 3, 4, 4, 4]

    def get_current(self):
        return [None, None, 0.02, None, None, None, None, 0.02, None]

class FakePyNEC:
    def fr_card(self, *args, **kargs):
        pass

    def xq_card(self, *args, **kargs):
        pass
    
    def get_structure_currents(self, freq_index):
        return FakeSC()


def mock_geometry(self):
    self.c = FakePyNEC()
    self.excitation_pairs = [(2, 1, 1+0j), (4, 2, 0.5+0j)]

@patch('antenna_designer.Antenna.geometry', new=mock_geometry)
def test_impedence_with_mock_Antenna():

    a = Antenna(Builder())
    zs = a.impedance()
    assert len(zs) == 2
    assert abs(zs[0]-50) < 0.001 and abs(zs[1]-25) < 0.001

    zs = a.impedance(sum_currents=1)
    assert len(zs) == 1
    assert abs(zs[0]-16.6666667) < 0.001
