from .. import AntennaBuilder
import math
from types import MappingProxyType

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'freq': 28.57,
    'base': 7,
    'length': 5.084,
    'slope': 0.604
  })

  def build_wires(self):
    eps = 0.05

    # (0.5*self.length)^2 == x^2+z^2
    # z = self.slope*x
    # (0.5*self.length)^2 == x^2*(1+self.slope^2)
    # 0.5*self.length == x*sqrt(1+self.slope^2)

    x = 0.5*self.length/math.sqrt(1+self.slope**2)
    z = self.slope*x

    n_seg0 = 21
    n_seg1 = 3

    tups = []
    tups.extend([((-x,  -z), (-eps, 0), n_seg0, None)])
    tups.extend([(( eps, 0), ( x,  -z), n_seg0, None)])
    tups.extend([((-eps, 0), ( eps, 0), n_seg1, 1+0j)])

    new_tups = []
    for (yoff, zoff) in [(0, self.base)]:
      new_tups.extend([((0, y0+yoff, z0+zoff), (0, y1+yoff, z1+zoff), ns, ex) for ((y0, z0), (y1, z1), ns, ex) in tups])

    return new_tups
