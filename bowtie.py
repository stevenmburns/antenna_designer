from antenna import AntennaBuilder
import math
from types import MappingProxyType

class BowtieBuilder(AntennaBuilder):
  default_params = MappingProxyType({
    'freq': 28.57,
    'slope_top': .658,
    'slope_bot': .512,
    'base': 7,
    'length_top': 5.771,
    'length_bot': 5.68,
    'del_y': 4,
    'del_z': 2
  })

  def build_wires(self):
    eps = 0.05

    n_seg0 = 21
    n_seg1 = 3

    def element(length, slope):
      # diag = sqrt(y^2 + (y*slope)^2) = y*sqrt(1+slope^2)
      # length/2 = diag + y*slope = y*(slope + sqrt(1+slope^2))

      y = 0.5*length/(slope + math.sqrt(1+slope**2))
      z = slope*y

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
      return tups

    tups_top = element(self.length_top, self.slope_top)
    tups_bot = element(self.length_bot, self.slope_bot)

    new_tups = []
    for yoff in (-self.del_y, self.del_y):
      zoff = self.base+self.del_z
      new_tups.extend([((0, y0+yoff, z0+zoff), (0, y1+yoff, z1+zoff), ns, ex) for ((y0, z0), (y1, z1), ns, ex) in tups_top])
      zoff = self.base-self.del_z
      new_tups.extend([((0, y0+yoff, z0+zoff), (0, y1+yoff, z1+zoff), ns, ex) for ((y0, z0), (y1, z1), ns, ex) in tups_bot])

    return new_tups

class BowtieSingleBuilder(AntennaBuilder):
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

