from .. import AntennaBuilder
import math
from types import MappingProxyType

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'freq': 28.57,
    'slope': .363,
    'base': 7,
    'length': 5.185
  })

  def build_wires(self):
    eps = 0.05

    # diag = sqrt(y^2 + (y*slope)^2) = y*sqrt(1+slope^2)
    # length/2 = diag + y*slope = y*(slope + sqrt(1+slope^2))

    y = 0.5*self.length/(self.slope + math.sqrt(1+self.slope**2))
    z = self.slope*y

    n_seg0 = 21
    n_seg1 = 3

    tups = []
    tups.extend([((-y,    0),   (-y,   z),    n_seg0, False)])
    tups.extend([((-y,    z),   (-eps, eps),  n_seg0, False)])
    tups.extend([((-eps,  eps), ( eps, eps),  n_seg1, False)])
    tups.extend([(( eps,  eps), ( y,   z),    n_seg0, False)])
    tups.extend([(( y,    z),   ( y,   0),    n_seg0, False)])
    tups.extend([((-y,    0),   (-y,   -z),   n_seg0, False)])
    tups.extend([((-y,   -z),   (-eps, -eps), n_seg0, False)])
    tups.extend([(( eps, -eps), ( y,   -z),   n_seg0, False)])
    tups.extend([(( y,   -z),   ( y,    0),   n_seg0, False)])
    tups.extend([((-eps, -eps), ( eps, -eps), n_seg1, True)])

    new_tups = []
    for (yoff, zoff) in [(0, self.base)]:
      new_tups.extend([((0, y0+yoff, z0+zoff), (0, y1+yoff, z1+zoff), ns, ex) for ((y0, z0), (y1, z1), ns, ex) in tups])

    return new_tups

