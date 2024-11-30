from ... import AntennaBuilder
import math
from types import MappingProxyType

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'freq': 28.57,
    'base': 7,
    'length_factor': 1,
    'slope': 0.604
  })

  def build_wires(self):
    eps = 0.05
    b = self.base

    # (0.5*self.length)^2 == x^2+z^2
    # z = self.slope*x
    # (0.5*self.length)^2 == x^2*(1+self.slope^2)
    # 0.5*self.length == x*sqrt(1+self.slope^2)

    wavelength = 300/self.freq

    y = 0.25*wavelength*self.length_factor/math.sqrt(1+self.slope**2)
    z = self.slope*y

    n_seg0 = 21
    n_seg1 = 3

    return [
      ((0, -y,  b-z), (0, -eps, b),  n_seg0, None),
      ((0,  eps, b),  (0,  y,  b-z), n_seg0, None),
      ((0, -eps, b),  (0,  eps, b),  n_seg1, 1+0j)
    ]
