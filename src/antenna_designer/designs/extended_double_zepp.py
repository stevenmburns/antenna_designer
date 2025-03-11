from antenna_designer import AntennaBuilder
import math
from math import sqrt
from types import MappingProxyType
# from icecream import ic

class Builder(AntennaBuilder):

  default_params = MappingProxyType({
    'freq': 28.57,
    'base': 7,
    'length': 13.125,
    'slope': 0.604,
  })


  def build_wires(self):
    eps = 0.05

    def build_path(lst, ns, ex):
      return ((a,b,ns,ex) for a,b in zip(lst[:-1], lst[1:]))
    def ry(p):
      return  p[0], -p[1], p[2]
    def rx(p):
      return  -p[0], p[1], p[2]

    S = (0, eps, 0) 
    T = ry(S)

    """
    r^2 = y^2 + z^2
    rho^2 = y^2
    y^2*slope^2 = z^2

    r^2 = y^2 + y^2 * slope^2
    r^2 = y^2*(1+slope^2)

    y^2 = r^2/(1+slope^2)

    z^2 = r^2 - y^2
"""

    def compute(length, slope):
      r_sq = (length / 2)**2
      y_sq = r_sq / (1+slope**2)
      z_sq = r_sq - y_sq

      return sqrt(y_sq), sqrt(z_sq)


    y, z = compute(self.length, self.slope)

    A = ( 0, eps+y, -z)
    B = ry(A)

    n_seg0 = 21
    n_seg1 = 1
      
    tups = []
    tups.append((S, A, n_seg0, None))
    tups.append((T, B, n_seg0, None))
    tups.append((T, S, n_seg1, 1+0j))

    new_tups = []
    for (xoff, yoff, zoff) in [(0, 0, self.base)]:
      new_tups.extend([((x0+xoff, y0+yoff, z0+zoff), (x1+xoff, y1+yoff, z1+zoff), ns, ev) for ((x0, y0, z0), (x1, y1, z1), ns, ev) in tups])

    return new_tups
