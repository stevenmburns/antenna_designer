from ..builder import Array1x2Builder
from .freq_based import hourglass

from types import MappingProxyType

class Builder(Array1x2Builder):
  default_params = MappingProxyType({
    'design_freq': 28.47,
    'freq': 28.47,
    'height_factor_top': 0.7493,
    'waist_factor_top': 0.3735,
    'width_factor_top': 0.9946,
    'base': 10,
    'del_y': 4,
    'del_z': 0,
  })

  def __init__(self, params = None):
    super().__init__(hourglass.Builder, params)
