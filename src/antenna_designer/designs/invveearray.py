from .. import Array2x2Builder
from . import invvee

from types import MappingProxyType

class Builder(Array2x2Builder):
  old_params = MappingProxyType({
    'freq': 28.47,
    'base': 7,
    'length_top': 5.084,
    'length_bot': 5.084,
    'slope_top': 0.604,
    'slope_bot': 0.604,
    'del_y': 4,
    'del_z': 2
  })

  default_params = MappingProxyType({
    'freq': 28.47,
    'base': 7,
    'length_top': 5.2418,
    'length_bot': 5.2766,
    'slope_top': 0.8540,
    'slope_bot': 0.8398,
    'del_y': 4,
    'del_z': 2.0
  })

  def __init__(self, params = None):
    super().__init__(invvee.Builder, params)

