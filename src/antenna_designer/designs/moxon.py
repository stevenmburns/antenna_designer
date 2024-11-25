from .. import AntennaBuilder
from types import MappingProxyType

class Builder(AntennaBuilder):
  original_params = MappingProxyType({
    'freq': 28.57,
    'base': 7,
    'halfdriver': (147.25 / 2 + 22 + 3/16)*0.0254,
    'aspect_ratio': (53 + 11/16) / 147.25,
    'tipspacer_factor': (4 + 1/16)/ (53 + 11/16),
    't0_factor': (22 + 3/16)/ (53 + 11/16)
  })

  default_params = MappingProxyType(
{'freq': 28.57, 'base': 7, 'halfdriver': 2.4597430629596713, 'aspect_ratio': 0.3646010186757216, 'tipspacer_factor': 0.07729647745945359, 't0_factor': 0.4078045966770739}
  )

  opt_params = MappingProxyType(
{'freq': 28.57, 'base': 7, 'halfdriver': 2.4454699666515394, 'aspect_ratio': 0.3646010186757216, 'tipspacer_factor': 0.047061074343758946, 't0_factor': 0.42268888502818136}
  )

  def build_wires(self):
    eps = 0.05

    """
    A = 147.25
    B = 22 3/16
    C = 4 1/16
    D = 27 7/16
    E = 53 11/16
"""

    # short = aspect_ratio*long
    # halfdriver = long/2 + short*t0_factor
    # halfdriver = long/2 + aspect_ratio*long*t0_factor
    # 2*halfdriver = long + 2*aspect_ratio*long*t0_factor
    # 2*halfdriver = long(1 + 2*aspect_ratio*t0_factor)
    # 2*halfdriver/(1 + 2*aspect_ratio*t0_factor) = long

    long = 2*self.halfdriver / (1 + 2*self.aspect_ratio*self.t0_factor)
    short = self.aspect_ratio * long

    tipspacer = short * self.tipspacer_factor
    t0 = short * self.t0_factor

    def build_path(lst, ns, ex):
      return ((a,b,ns,ex) for a,b in zip(lst[:-1], lst[1:]))

    def rx(p):
      return -p[0],  p[1], p[2]
    def ry(p):
      return  p[0], -p[1], p[2]

    S = (0, eps, 0) 
    T = ry(S)

    A = (0, long/2, 0)
    B = (A[0]-t0, A[1], 0)
    C = (B[0]-tipspacer, B[1], 0)
    D = (-short, long/2, 0)
    E = ry(D)
    F = ry(C)
    G = ry(B)
    H = ry(A)

    n_seg0 = 21
    n_seg1 = 1
      
    tups = []
    tups.extend(build_path([S,A,B], n_seg0, False))
    tups.extend(build_path([C,D,E,F], n_seg0, False))
    tups.extend(build_path([G,H,T], n_seg0, False))
    tups.append((T, S, n_seg1, True))

    new_tups = []
    for (xoff, yoff, zoff) in [(0, 0, self.base)]:
      new_tups.extend([((x0+xoff, y0+yoff, z0+zoff), (x1+xoff, y1+yoff, z1+zoff), ns, ex) for ((x0, y0, z0), (x1, y1, z1), ns, ex) in tups])

    return new_tups
