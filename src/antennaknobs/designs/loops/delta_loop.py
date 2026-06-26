import logging
import math
from types import MappingProxyType

from ... import AntennaBuilder

logger = logging.getLogger(__name__)


class Builder(AntennaBuilder):
    z100_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "base": 7.0,
            "length_factor": 1.0800,
            "angle_deg": 62.3894,
        }
    )

    z200_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "base": 7.0,
            "length_factor": 1.0650,
            "angle_deg": 43.9516,
        }
    )

    default_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "base": 7.0,
            "length_factor": 1.0800,
            "angle_deg": 62.3894,
        }
    )

    def build_wires(self):
        eps = 0.05
        b = self.base

        wavelength = 299.792458 / self.design_freq

        driver = wavelength * self.length_factor

        angle = math.radians(self.angle_deg)
        cos_theta = math.cos(angle)
        tan_theta = math.tan(angle)

        def build_path(lst, ns, ex):
            return ((a, b, ns, ex) for a, b in zip(lst[:-1], lst[1:]))

        def ry(p):
            return p[0], -p[1], p[2]

        def dist(p0, p1):
            return math.sqrt(sum((e0 - e1) ** 2 for e0, e1 in zip(p0, p1)))

        n_seg0 = self.nominal_nsegs
        n_seg1 = max(3, self.nominal_nsegs // 7)

        d = driver
        h = (cos_theta * (d - 2 * eps) + 2 * eps) / (2 * (cos_theta + 1))

        r"""
         B-----------------A
          \         theta /
           \             /
            \           /
             \         /
              \       /
               \     /
                T---S
    """

        S = (0, eps, b - (h - eps) * tan_theta)
        A = (0, h, b)

        B, T = ry(A), ry(S)

        logger.debug("theta = %.1f", angle * 180 / math.pi)
        logger.debug(
            "wires AB = %.3f AS = %.3f BT = %.3f",
            dist(A, B),
            dist(A, S),
            dist(B, T),
        )

        tups = []

        tups.extend(build_path([S, A, B, T], n_seg0, None))
        tups.extend(build_path([T, S], n_seg1, 1 + 0j))

        return tups
