from ...builder import Array1x2Builder
from ..loops import delta_loop

from types import MappingProxyType


class Builder(Array1x2Builder):
    default_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "length_factor_top": 1.0664,
            "angle_deg_top": 61.2377,
            "base": 7.0,
            "del_y": 4.0,
            "del_z": 0.0,
            "phase_lr": 0.0,
        }
    )

    dy35_dz2_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "length_factor_top": 1.0843,
            "angle_deg_top": 65.0422,
            "base": 7.0,
            "del_y": 3.5,
            "del_z": 2,
            "phase_lr": 0.0,
        }
    )

    dy45_dz2_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "length_factor_top": 1.0801,
            "angle_deg_top": 63.2603,
            "base": 7.0,
            "del_y": 4.5,
            "del_z": 2,
            "phase_lr": 0.0,
        }
    )

    dy3_dz2_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "length_factor_top": 1.0801,
            "angle_deg_top": 63.2603,
            "base": 7.0,
            "del_y": 3,
            "del_z": 2,
            "phase_lr": 0.0,
        }
    )

    def __init__(self, params=None):
        super().__init__(delta_loop.Builder, params)
