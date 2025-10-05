from .. import Array2x2Builder
from .freq_based import yagi

from types import MappingProxyType

class Builder(Array2x2Builder):
  default_params = MappingProxyType({
    'freq': 28.47,
    'base': 7,
    'length_factor_top': 0.9866,
    'length_factor_bot': 0.9866,
#    'angle_radians_top': 0,
#    'angle_radians_bot': 0,
    'angle_radians_top': 0.43,
    'angle_radians_bot': 0.43,
    'del_y': 4.0,
    'del_z': 2.0,
    'reflector_factor': 1.05,
    'boom_factor': 0.2,
    'n_directors': 2,
  })

  def __init__(self, params = None):
    super().__init__(yagi.Builder, params)

