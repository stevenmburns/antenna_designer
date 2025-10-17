from ... import AntennaBuilder
import math

from types import MappingProxyType

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'design_freq': 28.47,
    'freq': 28.47,
    'base': 7,
    'length_factor': 1.0724,
    'angle_radians': 0.8978,
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
    def rx(p):
      return  -p[0], p[1], p[2]
 
    n_seg0 = 21
    n_seg1 = 3


    d = driver
    h = d*cos_theta/4 - eps*cos_theta/2 + eps/2

    """
                  B 
                 / \
                /   \
               /     \
              /       \
             /         \
            /           \
           /             \
          /         theta \
         C           ------A
          \         theta /
           \             /
            \           /
             \         /
              \       /
               \     /
                T---S
    """

    B = (0, 0, b)
    A = (0, h, b-tan_theta*h)
    S = (0, eps, b-h*tan_theta-(h-eps)*tan_theta)
    C, T= ry(A), ry(S)

    tups = []

    tups.extend(build_path([S,A,B,C,T], n_seg0, None))
    tups.extend(build_path([T,S], n_seg1, 1+0j))

    B = (0, 0, eps+b)
    A = (h, 0, eps+b-tan_theta*h)
    S = (eps, 0, eps+b-h*tan_theta-(h-eps)*tan_theta)
    C, T= rx(A), rx(S)

    tups.extend(build_path([S,A,B,C,T], n_seg0, None))
    tups.extend(build_path([T,S], n_seg1, 1+0j))

    return tups
