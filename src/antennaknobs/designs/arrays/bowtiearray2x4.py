from ... import Array2x4Builder
from ..specialty import bowtie

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
            "phase_lr": 0.0,
            "phase_tb": 0.0,
            # Lay the panel out to mirror the array's physical structure on a
            # 4-column grid. The slope and length knobs are each a 2x2
            # (rows = top/bottom, cols = outer/inner) block — slopes on the
            # left half, matching lengths on the right — so the four element
            # shapes line up where they sit in the array. Geometry spacing
            # and feed phasing get their own rows underneath.
            "ui_params": MappingProxyType(
                {
                    "layout": {"columns": 4},
                    # slope block (cols 1-2: outer | inner)
                    "slope_otop": {"layout": {"row": 1, "col": 1}},
                    "slope_itop": {"layout": {"row": 1, "col": 2}},
                    "slope_obot": {"layout": {"row": 2, "col": 1}},
                    "slope_ibot": {"layout": {"row": 2, "col": 2}},
                    # length block (cols 3-4: outer | inner)
                    "length_otop": {"layout": {"row": 1, "col": 3}},
                    "length_itop": {"layout": {"row": 1, "col": 4}},
                    "length_obot": {"layout": {"row": 2, "col": 3}},
                    "length_ibot": {"layout": {"row": 2, "col": 4}},
                    # array spacing
                    "base": {"layout": {"row": 3, "col": 1}},
                    "del_y": {"layout": {"row": 3, "col": 2}},
                    "del_z": {"layout": {"row": 3, "col": 3}},
                    # feed phasing
                    "phase_lr": {"layout": {"row": 4, "col": 1}},
                    "phase_tb": {"layout": {"row": 4, "col": 2}},
                }
            ),
        }
    )

    def __init__(self, params=None):
        super().__init__(bowtie.Builder, params)
