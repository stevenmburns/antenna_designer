from ... import AntennaBuilder
import math

from types import MappingProxyType

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'freq': 28.47,
    'base': 7,
    'length_factor': 0.95,
    'director_factor': 0.95,
    'reflector_factor': 1.05,
    'boom_factor': 0.2,
    'angle_radians': 0.0,
#    'angle_radians': 0.4636476090008061,
    'n_directors': 2,
  })
  

  def build_wires(self):
    eps = 0.05
    b = self.base

    wavelength = 299.792458/self.freq

    driver_y = 0.25*wavelength*self.length_factor
    reflector_y = driver_y*self.reflector_factor
    director_y = driver_y*self.director_factor
    boom_x =  wavelength*self.boom_factor

    z_sin = math.sin(self.angle_radians)
    y_cos = math.cos(self.angle_radians)

    def build_path(lst, ns, ex):
      return ((a,b,ns,ex) for a,b in zip(lst[:-1], lst[1:]))
    def ry(p):
      return  p[0], -p[1], p[2]

    n_seg0 = 21
    n_seg1 = 3

    """
    B                    
    |                    A
    |                    |
    |                    |
    |                    |
    |                    |
    |                    |
    |                    |
    |                    S
    |                    |
    |                    T
    |                    |
    |                    |
    |                    |
    |                    |
    |                    |
    |                    D
    C
    """

    S = (0, eps,      b)
    A = (0, eps+(driver_y-eps)*y_cos, b-(driver_y-eps)*z_sin)
   
    U = (-boom_x, 0,  b)
    B = (-boom_x, reflector_y*y_cos, b-reflector_y*z_sin)

    C, D, T = ry(B), ry(A), ry(S)

    tups = []

    tups.extend(build_path([S,A], n_seg0, None))
    tups.extend(build_path([B,U,C], n_seg0, None))
    tups.extend(build_path([D,T], n_seg0, None))
    tups.extend(build_path([T,S], n_seg1, 1+0j))

    for i in range(self.n_directors):
      U = (boom_x*(1+i), 0,  b)
      B = (boom_x*(1+i), director_y*y_cos, b-director_y*z_sin)
      C = ry(B)
      tups.extend(build_path([B,U,C], n_seg0, None))

      

    return tups
