from .. import Array2x2Builder
from .freq_based import folded_invvee

from types import MappingProxyType

class Builder(Array2x2Builder):
  default_params = MappingProxyType({
    'freq': 28.47,
    'base': 7,
    'length_factor_top': 0.9943,
    'length_factor_bot': 1.0038,
    'angle_radians_top': 0.7255,
    'angle_radians_bot': 0.7246,
    'del_y': 4.0,
    'del_z': 2.0
  })

  def __init__(self, params = None):
    super().__init__(folded_invvee.Builder, params)

