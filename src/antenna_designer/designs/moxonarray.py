from .. import Array2x2Builder
from . import moxon

from types import MappingProxyType

class Builder(Array2x2Builder):
  default_params = MappingProxyType({
    'freq': 28.47,
    'base': 7,
    'del_y': 4,
    'del_z': 2,
    'halfdriver_top': 2.4515,
    'halfdriver_bot': 2.4487,
    'aspect_ratio_top': 0.3646010186757216,
    'aspect_ratio_bot': 0.3646010186757216,
    'tipspacer_factor_top': 0.07729647745945359,
    'tipspacer_factor_bot': 0.07729647745945359,
    't0_factor_top': 0.4078045966770739,
    't0_factor_bot': 0.4078045966770739

  })

  def __init__(self, params = None):
    super().__init__(moxon.Builder, params)

