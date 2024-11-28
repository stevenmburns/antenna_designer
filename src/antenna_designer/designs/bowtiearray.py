from .. import Array2x2Builder
from . import bowtie

from types import MappingProxyType

class Builder(Array2x2Builder):
  default_params = MappingProxyType({
    'freq': 28.57,
    'slope_top': .658,
    'slope_bot': .512,
    'base': 7,
    'length_top': 5.771,
    'length_bot': 5.68,
    'del_y': 4,
    'del_z': 2,
    'phase_lr': 0,
    'phase_tb': 0
  })

  def __init__(self, params = None):
    super().__init__(bowtie.Builder, params)
