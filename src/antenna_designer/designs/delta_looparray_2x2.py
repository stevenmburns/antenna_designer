from ..builder import Array2x2Builder
from .freq_based import delta_loop

from types import MappingProxyType

class Builder(Array2x2Builder):
  default_params = MappingProxyType({
    'design_freq': 28.47,
    'freq': 28.47,
    'length_factor_top': 1.1112,
    'angle_radians_top': 0.9111,
    'length_factor_bot': 1.1276,
    'angle_radians_bot': 0.9215,
    'base': 15,
    'del_y': 4,
    'del_z': 2,
  })

  def __init__(self, params = None):
    super().__init__(delta_loop.Builder, params)
