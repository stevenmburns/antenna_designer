from .. import Array2x4Builder
from . import bowtie

from types import MappingProxyType


class Builder(Array2x4Builder):
    default_params = MappingProxyType(
        {
            "freq": 28.57,
            "slope_itop": 0.658,
            "slope_ibot": 0.512,
            "slope_otop": 0.658,
            "slope_obot": 0.512,
            "base": 7.0,
            "length_itop": 5.771,
            "length_ibot": 5.68,
            "length_otop": 5.771,
            "length_obot": 5.68,
            "del_y": 4.0,
            "del_z": 2.0,
        }
    )

    def __init__(self, params=None):
        super().__init__(bowtie.Builder, params)
