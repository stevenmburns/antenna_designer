from ... import AntennaBuilder

from types import MappingProxyType

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'freq': 28.57,
    'base': 7,
    'length_factor': 1,
    'reflector_factor': 1.1,
    'boom_factor': 1
  })

  def build_wires(self):
    eps = 0.05
    b = self.base

    wavelength = 300/self.freq

    driver_y = 0.25*wavelength*self.length_factor
    reflector_y = 0.25*wavelength*self.reflector_factor
    boom_x =  0.25*wavelength*self.boom_factor

    def build_path(lst, ns, ex):
      return ((a,b,ns,ex) for a,b in zip(lst[:-1], lst[1:]))
    def rx(p):
      return -p[0],  p[1], p[2]
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
    A = (0, driver_y, b)
   
    B = (-boom_x, reflector_y, b)

    C, D, T = ry(B), ry(A), ry(S)

    tups = []

    tups.extend(build_path([S,A], n_seg0, None))
    tups.extend(build_path([B,C], n_seg0, None))
    tups.extend(build_path([D,T], n_seg0, None))
    tups.extend(build_path([T,S], n_seg1, 1+0j))

    return tups
