import math

from ... import AntennaBuilder
from ... import Transform, TransformStack

from types import MappingProxyType



class Builder(AntennaBuilder):
  params_50 = MappingProxyType({
    'design_freq': 28.47,
    'freq': 28.47,
    'base': 10,
    'top_height_factor': 0.4903,
    'mid_height_factor': 0.1042,
    'width_factor': 0.1577,   
    'slant_degrees': 30,
  })
  
  default_params = params_50

  def build_wires(self):
    eps = 0.05
    b = self.base

    wavelength = 299.792458/self.design_freq
 
    slant_radians = self.slant_degrees/180*math.pi
    cos_slant = math.cos(slant_radians)
    sin_slant = math.sin(slant_radians)

    def ry(p):
      return  p[0], -p[1], p[2]
    def rz(p):
      return  p[0], p[1], -p[2]

    n_seg0 = 21
    n_seg1 = 3

    """
    We will also model an invvee-like slant from the center for all three lines
    Zero degrees is horizonal
 C-------------AA-------------A
 |                            |
 |                            |
 |                            |
 |                            |
 |                            |
 |                            |
 |                            |
 D------------T--S------------B
 |                            |
 |                            |
 |                            |
 |                            |
 E-------------FF-------------F
    """

    S = (0, eps, wavelength*(self.mid_height_factor-self.top_height_factor))
    B = (0, wavelength*self.width_factor/2*cos_slant, wavelength*(self.mid_height_factor-self.top_height_factor)-wavelength*self.width_factor/2*sin_slant)
    A = (0, wavelength*self.width_factor/2*cos_slant, -wavelength*self.width_factor/2*sin_slant)
    AA = (0, 0, 0)

    F = (0, wavelength*self.width_factor/2*cos_slant, wavelength*(-self.top_height_factor)-wavelength*self.width_factor/2*sin_slant)
    FF = (0, 0, wavelength*(-self.top_height_factor))
   
    C, D, T = ry(A), ry(B), ry(S)
    E = ry(F)

    st = TransformStack()
    st.push(Transform.translate(0,0,b))

    def build_path(lst, ns, ex):
      return ((st.hit(a),st.hit(b),ns,ex) for a,b in zip(lst[:-1], lst[1:]))

    tups = []

    tups.extend(build_path([B,A,AA,C,D], n_seg0, None))
    tups.extend(build_path([B,F,FF,E,D], n_seg0, None))
    tups.extend(build_path([S,B], n_seg0, None))
    tups.extend(build_path([D,T], n_seg0, None))
    tups.extend(build_path([T,S], n_seg1, 1+0j))

    return tups
