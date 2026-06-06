from ... import AntennaBuilder
import math

from types import MappingProxyType

class Builder(AntennaBuilder):
  z100_params = MappingProxyType({
    'design_freq': 28.47,
    'freq': 28.47,
    'base': 7,
    'length_factor': 1.0800,
    'angle_radians': 1.0889,
  })

  z200_params = MappingProxyType({
    'design_freq': 28.47,
    'freq': 28.47,
    'base': 7,
    'length_factor': 1.0650,
    'angle_radians': 0.7671,
  })

  default_params = MappingProxyType({
    'design_freq': 28.47,
    'freq': 28.47,
    'base': 7,
    'length_factor': 1.0800,
    'angle_radians': 1.0889,
  })
  

  def build_wires(self):
    eps = 0.05
    b = self.base


    wavelength = 299.792458/self.design_freq

    driver = wavelength*self.length_factor

    cos_theta = math.cos(self.angle_radians)
    tan_theta = math.tan(self.angle_radians)

    def build_path(lst, ns, ex):
      return ((a,b,ns,ex) for a,b in zip(lst[:-1], lst[1:]))
    def ry(p):
      return  p[0], -p[1], p[2]
 
    def dist(p0, p1):
      return math.sqrt(sum((e0-e1)**2 for e0,e1 in zip(p0,p1)))

    n_seg0 = 21
    n_seg1 = 3


    d = driver
    h = (cos_theta*(d-2*eps)+2*eps)/(2*(cos_theta+1))

    """
         B-----------------A
          \         theta /
           \             /
            \           /
             \         /
              \       /
               \     /
                T---S
    """

    S = (0, eps, b-(h-eps)*tan_theta)
    A = (0, h, b)
   
    B, T= ry(A), ry(S)

    print(f'theta = {self.angle_radians*180/math.pi:.1f}')
    print(f'wires AB = {dist(A,B):.3f} AS = {dist(A,S):.3f} BT ={dist(B,T):.3f}')

    tups = []

    tups.extend(build_path([S,A,B,T], n_seg0, None))
    tups.extend(build_path([T,S], n_seg1, 1+0j))

    return tups
