from ... import AntennaBuilder
import math

from ... import Transform, TransformStack

import numpy as np

from types import MappingProxyType

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'design_freq': 28.47,
    'freq': 28.47,
    'base': 7,
    'length_factor': 1.0664,
    'angle_radians': 1.0688,
    'slant': 0,
    'twist': 0.125,
    'del_y': 4,
  })
  

  def build_tls(self):
    return self.tls

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

    """
    """

    st = TransformStack()
    st.push(Transform.translate(0,0,b))
    st.push(Transform.rotX(-self.slant))
    st.push(Transform.translate(0,self.del_y,-b))


    SS, AA, BB, TT = st.hit(S), st.hit(A), st.hit(B), st.hit(T)

    SSS, AAA, BBB, TTT= ry(SS), ry(AA), ry(BB), ry(TT)

    tups = []

    tups.extend(build_path([SS,AA,BB,TT], n_seg0, None))
    tups.extend(build_path([TT,SS], n_seg1, 1+0j))

    tups.extend(build_path([SSS,AAA,BBB,TTT], n_seg0, None))
    tups.extend(build_path([SSS,TTT], n_seg1, 1+0j))


    WW  = (SS[0], eps, SS[1])
    WWW = ry(WW)

    self.tls = []

    tups.extend(build_path([WWW,WW], n_seg1, 1+0j))

    feedpoints = [(idx, x) for idx, x in enumerate(tups, start=1) if x[3] is not None]

    assert len(feedpoints) == 3

    tl_lengths = self.del_y - wavelength*self.twist, self.del_y + wavelength*self.twist


    for (idx, (p0, p1, nsegs, ev)), tl_length in zip(feedpoints[:2], tl_lengths):
      self.tls.append( (idx, (n_seg1+1)//2, len(tups), (n_seg1+1)//2, 100, tl_length))
      tups[idx-1] = (p0, p1, nsegs, None)

    return tups
  
