from ..builder import Array1x2Builder
from . import bowtie

from types import MappingProxyType

class Builder(Array1x2Builder):
  default_params = MappingProxyType({
    'freq': 28.47,
    'slope_top': .580,
    'base': 7,
    'length_top': 5.53,
    'del_y': 4,
    'del_z': 2,
  })

  def __init__(self, params = None):
    super().__init__(bowtie.Builder, params)
