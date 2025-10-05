from .. import AntennaBuilder
import math
from types import MappingProxyType

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'freq': 28.47,
    'base': 7,
    'length': 5.09,
    'slope': 0.5,
    'excitation': 1+0j
  })

  def build_wires(self):
    eps = 0.05
    b = self.base

    # (0.5*self.length)^2 == y^2+z^2
    # z = self.slope*y
    # (0.5*self.length)^2 == y^2*(1+self.slope^2)
    # 0.5*self.length == y*sqrt(1+self.slope^2)

    y = 0.5*self.length/math.sqrt(1+self.slope**2)
    z = self.slope*y

    n_seg0 = 21
    n_seg1 = 3

    return (
      ((0, -y,   b-z), (0, -eps, b),   n_seg0, None),
      ((0, eps,  b),   (0, y,    b-z), n_seg0, None),
      ((0, -eps, b),   (0, eps,  b),   n_seg1, self.excitation)
    )
