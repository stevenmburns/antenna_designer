from .. import AntennaBuilder

import math
from types import MappingProxyType

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'freq': 28.57,
    'length': 2.619,
    'base': .5
  })

  def build_wires(self):
    eps = 0.05

    z = self.length

    n_seg0 = 21
    n_seg1 = 3
    n_seg_radials = 5

    tups = []
    tups.extend([((0, 0, 0),   (0, 0, eps), n_seg1, 1+0j)])
    tups.extend([((0, 0, eps), (0, 0, z),   n_seg0, None)])

    n_radials = 3

    for i in range(n_radials):
      theta = 2*math.pi/n_radials*i

      x, y = self.length*math.cos(theta), self.length*math.sin(theta)

      tups.extend([((0, 0, 0), (x, y, 0), n_seg_radials, None)])

    base = self.base
    new_tups = []
    for (x0, y0, z0), (x1, y1, z1), n_seg, ev in tups:
      new_tups.append(( (x0, y0, z0+base), (x1, y1, z1+base), n_seg, ev))

    return new_tups

