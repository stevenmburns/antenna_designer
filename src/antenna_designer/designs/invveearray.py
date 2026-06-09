from .. import Array2x2Builder
from .freq_based import invvee

from types import MappingProxyType


class Builder(Array2x2Builder):
    # Pre-2024 params with the looser slope=0.604 and length=5.084 each;
    # kept for backwards-compat tuning loads. length_factor / angle_radians
    # are the freq_based.invvee-equivalent values:
    #   length_factor = (length / 2) / (0.25 · λ_design)
    #                 = 2.542 / 2.6325 = 0.9656
    #   angle_radians = atan(slope) = atan(0.604) = 0.5434
    old_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "base": 7.0,
            "length_factor_top": 0.9656,
            "length_factor_bot": 0.9656,
            "angle_radians_top": 0.5434,
            "angle_radians_bot": 0.5434,
            "del_y": 4.0,
            "del_z": 2,
            "phase_lr": 0.0,
            "phase_tb": 0.0,
        }
    )

    # Tuned 2024 params. Same converted-from-top-level values:
    #   length_top=5.2418, slope_top=0.854 → length_factor=0.9956, angle=0.7068
    #   length_bot=5.2766, slope_bot=0.840 → length_factor=1.0022, angle=0.6985
    default_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "base": 7.0,
            "length_factor_top": 0.9956,
            "length_factor_bot": 1.0022,
            "angle_radians_top": 0.7068,
            "angle_radians_bot": 0.6985,
            "del_y": 4.0,
            "del_z": 2.0,
            "phase_lr": 0.0,
            "phase_tb": 0.0,
        }
    )

    def __init__(self, params=None):
        super().__init__(invvee.Builder, params)
