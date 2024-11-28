from antenna_designer import AntennaBuilder

from types import MappingProxyType

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'freq': 28.57,
    'base': 7,
    'length_factor': 1
  })

  def build_wires(self):
    eps = 0.05
    b = self.base

    wavelength = 300/self.freq

    y = 0.25*wavelength*self.length_factor

    n_seg0 = 21
    n_seg1 = 3

    return [
      ((0, -y,   b),  (0, -eps, b), n_seg0, None),
      ((0,  eps, b),  (0,  y,   b), n_seg0, None),
      ((0, -eps, b),  (0,  eps, b), n_seg1, 1+0j)
    ]

class FancyBuilder(AntennaBuilder):
  default_params = MappingProxyType({
    'freq': 28.57,
    'base': 7,
    'length_factor': 1
  })

  def build_wires(self):
    eps = 0.05
    b = self.base

    wavelength = 300/self.freq

    y = 0.25*wavelength*self.length_factor

    n_seg0 = 21
    n_seg1 = 3

    return [
      ((0, -y,   b),  (0, -eps, b), n_seg0, None),
      ((0,  eps, b),  (0,  y,   b), n_seg0, None),
      ((0, -eps, b),  (0,  eps, b), n_seg1, 1+0j)
    ]
