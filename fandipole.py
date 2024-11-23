from antenna import AntennaBuilder
import math
from types import MappingProxyType
# from icecream import ic

class FandipoleBuilder(AntennaBuilder):
  old_params = MappingProxyType({
    'freq': 28.57,
    'base': 7,
    'length_20': 5.084 * 20 / 10,
    'length_17': 5.084 * 17 / 10,
    'length_15': 5.084 * 15 / 10,
    'length_12': 5.084 * 12 / 10,
    'length_10': 5.084,
    'slope': 0.604
  })

  default_params0 = MappingProxyType({
    'freq': 28.57,
    'base': 7,
    'length_20': 11.3,
    'length_17': 11.3 * 17 / 20,
    'length_15': 11.3 * 15 / 20,
    'length_12': 11.3 * 12 / 20,
    'length_10': 11.3 * 10 / 20,
    'slope': 0.604
  })

  default_params = MappingProxyType({
    'freq': 28.57,
    'base': 7,
    'length_20': 10,
    'length_17': 10 * 14.3 / 18.157,
    'length_15': 10 * 14.3 / 21.383,
    'length_12': 10 * 14.3 / 24.97,
    'length_10': 10 * 14.3 / 28.57,
    'slope': 0.604
  })

  def build_wires(self):
    eps = 0.05

    radius = .3
    t0 = .5

    n = 2

    lst = [(math.cos(math.pi*i/180),math.sin(math.pi*i/180)) for i in range(360//(2*n), 360, 360//n)]
        
    def build_path(lst, ns, ex):
      return ((a,b,ns,ex) for a,b in zip(lst[:-1], lst[1:]))

    def rx(p):
      return -p[0],  p[1], p[2]
    def ry(p):
      return  p[0], -p[1], p[2]

    S = (0, eps, 0) 
    T = ry(S)

    C = (0, S[1]+t0*math.sqrt(1+self.slope**2), -self.slope*t0*math.sqrt(1+self.slope**2))

    A = [(C[0]+radius*x,
          C[1]+radius*y*self.slope*math.sqrt(1+self.slope**2),
          C[2]+radius*y*math.sqrt(1+self.slope**2)
    ) for (x, y) in lst]


    ls = [q/2 for q in [self.length_10, self.length_12, self.length_15, self.length_17, self.length_20]]

    Ds = [(0, S[1]+q*math.sqrt(1+self.slope**2), -self.slope*q*math.sqrt(1+self.slope**2)) for q in ls]

    B = [(D[0]+radius*x,
          D[1]+radius*y*self.slope*math.sqrt(1+self.slope**2),
          D[2]+radius*y*math.sqrt(1+self.slope**2)
    ) for (x, y), D in zip(lst, Ds)]

    Ay = [ry(p) for p in A]
    By = [ry(p) for p in B]

    n_seg0 = 21
    n_seg1 = 1
      
    tups = []
    for i in range(n):
        tups.extend(build_path([S,A[i],B[i]], n_seg0, False))
        tups.extend(build_path([T,Ay[i],By[i]], n_seg0, False))
    tups.append((T, S, n_seg1, True))

    new_tups = []
    for (xoff, yoff, zoff) in [(0, 0, self.base)]:
      new_tups.extend([((x0+xoff, y0+yoff, z0+zoff), (x1+xoff, y1+yoff, z1+zoff), ns, ex) for ((x0, y0, z0), (x1, y1, z1), ns, ex) in tups])

    return new_tups
