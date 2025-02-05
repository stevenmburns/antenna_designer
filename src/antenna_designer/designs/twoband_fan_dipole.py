from antenna_designer import AntennaBuilder
import math
from math import sqrt
from types import MappingProxyType
# from icecream import ic

class Builder(AntennaBuilder):

  default_params = MappingProxyType({
    'freq': 28.57,
    'freq_10': 28.57,
    'freq_12': 24.97,
    'base': 7,
    'length_12': 5.769,
    'length_10': 4.9074,
    'slope': .5,
    'gap_slope': .33
  })

  """

       xx
         xx
           xx
             xx
             xx
           xx
         xx
       xx
     xx
   xx

"""


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
    r^2 = x^2 + y^2 + z^2
    rho^2 = x^2 + y^2
    (x^2 + y^2)*slope^2 = z^2
    rho^2*slope^2 = z^2

    r^2 = rho^2 + rho^2 * slope^2
    r^2 = rho^2*(1+slope^2)

    rho^2 = r^2/(1+slope^2)

    rho^2 = x^2 + y^2
    x*gap_slope= y

    rho^2 = x^2 + y^2
    x^2*gap_slope^2 = y^2
    rho^2 = x^2 + x^2*gap_slope^2
    rho^2 = x^2*(1+gap_slope^2)
    
    x^2 = rho^2/(1+gap_slope^2)

    y^2 = rho^2 - x^2
"""

    def compute(length, slope, gap_slope):
      r_sq = (length / 2)**2
      rho_sq = r_sq / (1+slope**2)
      z_sq = r_sq - rho_sq
      y_sq = rho_sq / (1+gap_slope**2)
      x_sq = rho_sq - y_sq

      return sqrt(x_sq), sqrt(y_sq), sqrt(z_sq)


    x_12, y_12, z_12 = compute(self.length_12, self.slope, self.gap_slope)
    x_10, y_10, z_10 = compute(self.length_10, self.slope, self.gap_slope)

    A = ( x_12, eps+y_12, -z_12)
    B = (-x_10, eps+y_10, -z_10)

    C = rx(ry(A))
    D = rx(ry(B))

    n_seg0 = 21
    n_seg1 = 1
      
    tups = []
    tups.append((S, A, n_seg0, None))
    tups.append((S, B, n_seg0, None))
    tups.append((T, C, n_seg0, None))
    tups.append((T, D, n_seg0, None))
    tups.append((T, S, n_seg1, 1+0j))

    new_tups = []
    for (xoff, yoff, zoff) in [(0, 0, self.base)]:
      new_tups.extend([((x0+xoff, y0+yoff, z0+zoff), (x1+xoff, y1+yoff, z1+zoff), ns, ev) for ((x0, y0, z0), (x1, y1, z1), ns, ev) in tups])

    return new_tups
