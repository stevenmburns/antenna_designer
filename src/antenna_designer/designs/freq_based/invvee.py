from ... import AntennaBuilder
import math

from types import MappingProxyType


class Builder(AntennaBuilder):
    default_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "base": 7.0,
            "length_factor": 0.9719,
            #    'angle_radians': 0.0,
            "angle_radians": 0.5530,
            # length_factor span has to cover both the half-wave default
            # (~0.97) and the EDZ variant (~2.97, encoding a ~1.5λ
            # element). Auto-derive's ±50% window would clip at ~1.46.
            "ui_params": MappingProxyType(
                {
                    "length_factor": {
                        "min": 0.4,
                        "max": 3.2,
                        "step": 0.001,
                        "precision": 4,
                    },
                }
            ),
        }
    )

    # Extended Double Zepp: same V-dipole geometry as the half-wave
    # default, just tuned to a ~1.5λ element. length_factor parameterises
    # driver_y as 0.25·λ·length_factor, so the EDZ's 0.7422·λ driver_y
    # lands at length_factor = 0.7422 / 0.25 = 2.9688. angle_radians=0
    # makes it a straight (flat) double Zepp; bend it down for an
    # inverted-V Zepp by raising angle_radians.
    edz_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "base": 7.0,
            "length_factor": 2.9688,
            "angle_radians": 0.0,
        }
    )

    def build_wires(self):
        eps = 0.05
        b = self.base

        wavelength = 299.792458 / self.design_freq

        driver_y = 0.25 * wavelength * self.length_factor

        z_sin = math.sin(self.angle_radians)
        y_cos = math.cos(self.angle_radians)

        def build_path(lst, ns, ex):
            return ((a, b, ns, ex) for a, b in zip(lst[:-1], lst[1:]))

        def ry(p):
            return p[0], -p[1], p[2]

        n_seg0 = 21
        n_seg1 = 3

        """
                    
                A
                |
                |
                |
                |
                |
                |
                |
                S
                |
                T
                |
                |
                |
                |
                |
                |
                |
                D

    """

        S = (0, eps, b)
        A = (0, eps + (driver_y - eps) * y_cos, b - (driver_y - eps) * z_sin)

        D, T = ry(A), ry(S)

        tups = []

        tups.extend(build_path([S, A], n_seg0, None))
        tups.extend(build_path([D, T], n_seg0, None))
        tups.extend(build_path([T, S], n_seg1, 1 + 0j))

        return tups
