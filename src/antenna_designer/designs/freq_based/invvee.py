from ... import AntennaBuilder
import math

from types import MappingProxyType

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'design_freq': 28.47,
    'freq': 28.47,
    'base': 7,
    'length_factor': 0.9719,
#    'angle_radians': 0.0,
    'angle_radians': 0.5530,
  })
  

  def build_wires(self):
    eps = 0.05
    b = self.base


    wavelength = 299.792458/self.design_freq

    driver_y = 0.25*wavelength*self.length_factor

    z_sin = math.sin(self.angle_radians)
    y_cos = math.cos(self.angle_radians)

    def build_path(lst, ns, ex):
      return ((a,b,ns,ex) for a,b in zip(lst[:-1], lst[1:]))
    def ry(p):
      return  p[0], -p[1], p[2]

    n_seg0 = 21
    n_seg1 = 3

    """
                    
                A
                |
                |
                |
                |
                |
                |
                |
                S
                |
                T
                |
                |
                |
                |
                |
                |
                |
                D

    """

    S = (0,          eps,      b)
    A = (0, eps+(driver_y-eps)*y_cos, b-(driver_y-eps)*z_sin)
   
    D, T = ry(A), ry(S)

    tups = []

    tups.extend(build_path([S,A], n_seg0, None))
    tups.extend(build_path([D,T], n_seg0, None))
    tups.extend(build_path([T,S], n_seg1, 1+0j))

    return tups
