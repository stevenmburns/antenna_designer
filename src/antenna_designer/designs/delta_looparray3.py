from ..builder import Array1x2Builder
from .freq_based import delta_loop

from types import MappingProxyType

class Builder(Array1x2Builder):
  default_params = MappingProxyType({
    'design_freq': 28.47,
    'freq': 28.47,
    'length_factor_top': 1.0801,
    'angle_radians_top': 1.1041,
    'base': 7,
    'del_y': 4.5,
    'del_z': 2,
  })

  def __init__(self, params = None):
    super().__init__(delta_loop.Builder, params)
