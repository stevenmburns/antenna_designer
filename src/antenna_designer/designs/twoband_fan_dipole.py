from antenna_designer import AntennaBuilder
import math
from math import sqrt
from types import MappingProxyType
# from icecream import ic

class Builder(AntennaBuilder):

  default_params_07 = MappingProxyType({
    'freq': 28.57,
    'freq_10': 28.57,
    'freq_12': 24.97,
    'base': 7,
    'length_12': 5.1102,
    'length_10': 4.4682,
    'slope_12': .5,
    'slope_10': .5,
    'gap_slope': 0,
    's': 0.7,
    'eps': 0.01
  })



  default_params_05 = MappingProxyType({
    'freq': 28.57,
    'freq_10': 28.57,
    'freq_12': 24.97,
    'base': 7,
    'length_12': 5.2949,
    'length_10': 4.6531,
    'slope_12': .5,
    'slope_10': .5,
    'gap_slope': 0,
    's': 0.5,
    'eps': 0.01
  })

  default_params_03 = MappingProxyType({
    'freq': 28.57,
    'freq_10': 28.57,
    'freq_12': 24.97,
    'base': 7,
    'length_12': 5.4725,
    'length_10': 4.8370,
    'slope_12': .5,
    'slope_10': .5,
    'gap_slope': 0,
    's': 0.3,
    'eps': 0.01
  })

  default_params_025 = MappingProxyType({
    'freq': 24.97,
    'freq_10': 28.57,
    'freq_12': 24.97,
    'base': 7,
    'length_12': 5.5153,
    'length_10': 4.8837,
    'slope_12': .5,
    'slope_10': .5,
    'gap_slope': 0,
    's': 0.25,
    'eps': 0.01
  })


  default_params_02 = MappingProxyType({
    'freq': 24.97,
    'freq_10': 28.57,
    'freq_12': 24.97,
    'base': 7,
    'length_12': 5.5571,
    'length_10': 4.9312,
    'slope_12': .5,
    'slope_10': .5,
    'gap_slope': 0,
    's': 0.2,
    'eps': 0.01
  })

  default_params_015 = MappingProxyType({
    'freq': 24.97,
    'freq_10': 28.57,
    'freq_12': 24.97,
    'base': 7,
    'length_12': 5.5978,
    'length_10': 4.9803,
    'slope_12': .5,
    'slope_10': .5,
    'gap_slope': 0,
    's': 0.15,
    'eps': 0.01
  })

  default_params_01 = MappingProxyType({
    'freq': 24.97,
    'freq_10': 28.57,
    'freq_12': 24.97,
    'base': 7,
    'length_12': 5.6371,
    'length_10': 5.0331,
    'slope_12': .5,
    'slope_10': .5,
    'gap_slope': 0,
    's': 0.10,
    'eps': 0.01
  })

  default_params_01_001 = MappingProxyType({
    'freq': 24.97,
    'freq_10': 28.57,
    'freq_12': 24.97,
    'base': 7,
    'length_12': 5.7628,
    'length_10': 5.0717,
    'slope_12': .5,
    'slope_10': .5,
    'gap_slope': 0,
    's': 0.10,
    'eps': 0.001
  })

  default_params_current_physical = MappingProxyType({
    'freq': 28.47,
    'freq_10': 29.3,
    'freq_12': 26.6,
    'base': 7,
    'length_12': 5.494,
    'length_10': 5.0517,
    'slope_12': .5,
    'slope_10': .5,
    'gap_slope': 0,
    's': 0.15,
    'eps': 0.015
  })

#
# Need to add 19.15 cm to the 12m band
#

#
# Need to add 7.13 cm to the 10m band
#


#
# Need to add 19.15 cm to the 12m band
#

#
# Need to add 7.13 cm to the 10m band
#


  default_params = MappingProxyType({
    'freq': 28.47,
    'freq_10': 28.47,
    'freq_12': 24.97,
    'base': 7,
    'length_12': 5.8770,
    'length_10': 5.1944,
    'slope_12': .5,
    'slope_10': .5,
    'gap_slope': 0,
    's': 0.15,
    'eps': 0.015
  })

  default_params = MappingProxyType({
    'freq': 28.47,
    'freq_10': 28.7,
    'freq_12': 24.0,
    'base': 7,
    'length_12': 5.877,
    'length_10': 5.19,
    'slope_12': .5,
    'slope_10': .5,
    'gap_slope': 0,
    's': 0.15,
    'eps': 0.015
  })


  # invvee reference 5.8408


  def build_wires(self):
    eps = 0.01


    def build_path(lst, ns, ex):
      return ((a,b,ns,ex) for a,b in zip(lst[:-1], lst[1:]))
    def ry(p):
      return  p[0], -p[1], p[2]
    def rx(p):
      return  -p[0], p[1], p[2]

    """
    r^2 = x^2 + y^2 + z^2
    rho^2 = x^2 + y^2
    (x^2 + y^2)*slope^2 = z^2
    rho^2*slope^2 = z^2

    r^2 = rho^2 + rho^2 * slope^2
    r^2 = rho^2*(1+slope^2)

    rho^2 = r^2/(1+slope^2)

    rho^2 = x^2 + y^2
    y*gap_slope = x

    rho^2 = x^2 + y^2
    y^2*gap_slope^2 = x^2
    rho^2 = y^2 + y^2*gap_slope^2
    rho^2 = y^2*(1+gap_slope^2)
    
    y^2 = rho^2/(1+gap_slope^2)

    x^2 = rho^2 - y^2
"""

    def compute(length, slope, gap_slope):
      r_sq = (length / 2)**2
      rho_sq = r_sq / (1+slope**2)
      z_sq = r_sq - rho_sq
      y_sq = rho_sq / (1+gap_slope**2)
      x_sq = rho_sq - y_sq

      return sqrt(x_sq), sqrt(y_sq), sqrt(z_sq)


    x_12, y_12, z_12 = compute(self.length_12-2*self.s, self.slope_12, self.gap_slope)
    x_10, y_10, z_10 = compute(self.length_10-2*self.s, self.slope_10, self.gap_slope)

    S = (0, eps, 0) 
    T = ry(S)

    G = (self.s/sqrt(2.0), eps+self.s/sqrt(2.0), 0)
    H = rx(G)

    I = ry(G)
    J = ry(H)

    A = ( x_12+G[0], y_12+G[1], -z_12)
    B = (-x_10+H[0], y_10+H[1], -z_10)

    C = ry(A)
    D = ry(B)

    n_seg0 = 21
    n_seg1 = 1

    def dist(p0, p1):
      return math.sqrt(sum((x0-x1)**2 for x0, x1 in zip(p0, p1)))
    

    wire12 = dist(S, G) + dist(G, A)
    wire10 = dist(S, H) + dist(H, B)

    print( f"wire12: {wire12} {self.length_12/2} wire10: {wire10} {self.length_10/2} ")


    tups = []
    tups.append((S, G, 5, None))
    tups.append((S, H, 5, None))
    tups.append((G, A, n_seg0, None))
    tups.append((H, B, n_seg0, None))

    tups.append((T, I, 5, None))
    tups.append((T, J, 5, None))
    tups.append((I, C, n_seg0, None))
    tups.append((J, D, n_seg0, None))

    tups.append((T, S, n_seg1, 1+0j))

    new_tups = []
    for (xoff, yoff, zoff) in [(0, 0, self.base)]:
      new_tups.extend([((x0+xoff, y0+yoff, z0+zoff), (x1+xoff, y1+yoff, z1+zoff), ns, ev) for ((x0, y0, z0), (x1, y1, z1), ns, ev) in tups])

    return new_tups



if __name__ == "__main__":

  def tofeet_inches(m):
    f, i = divmod(m/0.0254, 12)

    ii, frac16 = divmod(i*16, 16)

    frac16 = int(frac16+0.5)

    g = math.gcd(frac16, 16)

    return f"{m*100:.1f} cm {f:.0f} ft {i:.3f} in ({ii:.0f} {frac16//g}/{16//g} in)"

  params = Builder.default_params
  print(f"Quarter wave element on 12m: {tofeet_inches(params['length_12']/2)}")
  print(f"Quarter wave element on 10m: {tofeet_inches(params['length_10']/2)}")

  print(f"Ratio of 12m element to single band invvee: {params['length_12']/5.8408}")
  print(f"Ratio of 12m element to 10m element: {params['length_12']/params['length_10']}")  
