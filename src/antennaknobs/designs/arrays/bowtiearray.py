from ... import Array2x2Builder
from ..specialty import bowtie

from types import MappingProxyType


class Builder(Array2x2Builder):
    default_params = MappingProxyType(
        {
            "freq": 28.47,
            "slope_top": 0.658,
            "slope_bot": 0.512,
            "base": 7.0,
            "length_top": 5.79,
            "length_bot": 5.70,
            "del_y": 4.0,
            "del_z": 2.0,
            "phase_lr": 0.0,
            "phase_tb": 0.0,
            # slope / length as a top/bottom matrix in cols 1-2 (the 2x2
            # sibling, bowtiearray2x4, adds inner/outer columns); array
            # spacing on row 3, feed phasing on row 4.
            "ui_params": MappingProxyType(
                {
                    "layout": {"columns": 3},
                    "slope_top": {"layout": {"row": 1, "col": 1}},
                    "length_top": {"layout": {"row": 1, "col": 2}},
                    "slope_bot": {"layout": {"row": 2, "col": 1}},
                    "length_bot": {"layout": {"row": 2, "col": 2}},
                    "base": {"layout": {"row": 3, "col": 1}},
                    "del_y": {"layout": {"row": 3, "col": 2}},
                    "del_z": {"layout": {"row": 3, "col": 3}},
                    "phase_lr": {"layout": {"row": 4, "col": 1}},
                    "phase_tb": {"layout": {"row": 4, "col": 2}},
                }
            ),
        }
    )

    def __init__(self, params=None):
        super().__init__(bowtie.Builder, params)
