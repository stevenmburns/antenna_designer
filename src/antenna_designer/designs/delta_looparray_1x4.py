from ..builder import Array1x4Builder
from .freq_based import delta_loop

from types import MappingProxyType

class Builder(Array1x4Builder):
  default_params = MappingProxyType({
    'design_freq': 28.47,
    'freq': 28.47,
    'length_factor_itop': 1.0912,
    'angle_radians_itop': 0.9110,
    'length_factor_otop': 1.0795,
    'angle_radians_otop': 0.8911,
    'base': 7,
    'del_y': 4,
    'del_z': 2,
  })

  def __init__(self, params = None):
    super().__init__(delta_loop.Builder, params)
