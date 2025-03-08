from antenna_designer import AntennaBuilder
import math
from types import MappingProxyType
# from icecream import ic

class Builder(AntennaBuilder):
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
    'length_20': 8.1437,
    'length_17': 6.5164,
    'length_15': 5.5075,
    'length_12': 4.6646,
    'length_10': 4.1186,
    'freq_20': 14.300,
    'freq_17': 18.1575,
    'freq_15': 21.383,
    'freq_12': 24.97,
    'freq_10': 28.47,
    'slope': 0.5
  })

  def build_wires(self):
    eps = 0.01

    radius = .1
    t0 = radius / math.sqrt(2)

    n = 5

    lst = [(math.cos(math.pi*i/180),math.sin(math.pi*i/180)) for i in range(360//(2*n), 360, 360//n)]
        
    def build_path(lst, ns, ex):
      return ((a,b,ns,ex) for a,b in zip(lst[:-1], lst[1:]))
    def ry(p):
      return  p[0], -p[1], p[2]

    S = (0, eps, 0) 
    T = ry(S)

    C = (0, S[1]+t0*math.sqrt(1+self.slope**2), -self.slope*t0*math.sqrt(1+self.slope**2))

    A = [(C[0]+radius*x,
          C[1]+radius*y*self.slope*math.sqrt(1+self.slope**2),
          C[2]+radius*y*math.sqrt(1+self.slope**2)
    ) for (x, y) in lst]

    def dist(p0, p1):
      return math.sqrt(sum((x0-x1)**2 for x0, x1 in zip(p0, p1)))

    
    print(f"radius: {radius} dists: {[dist(S, a) for a in A]}")

    lengths = [self.length_10, self.length_12, self.length_15, self.length_17, self.length_20]

    ls = [(q/2 - dist(S, a)) for (q,a) in zip(lengths, A)]

    Ds = [(0, S[1]+q*math.sqrt(1+self.slope**2), -self.slope*q*math.sqrt(1+self.slope**2)) for q in ls]

    B = [(D[0]+radius*x,
          D[1]+radius*y*self.slope*math.sqrt(1+self.slope**2),
          D[2]+radius*y*math.sqrt(1+self.slope**2)
    ) for (x, y), D in zip(lst, Ds)]

    Ay = [ry(p) for p in A]
    By = [ry(p) for p in B]

    
    for i in range(n):
      wire_length = dist(S, A[i]) + dist(A[i], B[i])

      print( f"{i} length {wire_length} {lengths[i]/2} {(wire_length-lengths[i]/2)/lengths[i]}")


    n_seg0 = 21
    n_seg1 = 1
      
    tups = []
    for i in range(n):
      tups.extend(build_path([S,A[i],B[i]], n_seg0, None))
      tups.extend(build_path([T,Ay[i],By[i]], n_seg0, None))
    tups.append((T, S, n_seg1, 1+0j))

    new_tups = []
    for (xoff, yoff, zoff) in [(0, 0, self.base)]:
      new_tups.extend([((x0+xoff, y0+yoff, z0+zoff), (x1+xoff, y1+yoff, z1+zoff), ns, ev) for ((x0, y0, z0), (x1, y1, z1), ns, ev) in tups])

    return new_tups
