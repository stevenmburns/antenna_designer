from ... import AntennaBuilder
import math
from types import MappingProxyType


class Builder(AntennaBuilder):
    default_params = MappingProxyType(
        {"freq": 28.47, "angle_deg": 28.2625, "base": 9.0, "length": 5.4213}
    )

    def build_wires(self):
        eps = 0.05

        # angle_deg is the arm droop angle; slope = tan(droop angle) = dz/dy.
        # diag = sqrt(y^2 + (y*slope)^2) = y*sqrt(1+slope^2)
        # length/2 = diag + y*slope = y*(slope + sqrt(1+slope^2))

        slope = math.tan(math.radians(self.angle_deg))
        y = 0.5 * self.length / (slope + math.sqrt(1 + slope**2))
        z = slope * y

        n_seg0 = self.nominal_nsegs
        n_seg1 = max(3, self.nominal_nsegs // 7)

        tups = []
        tups.extend([((-y, 0), (-y, z), n_seg0, None)])
        tups.extend([((-y, z), (-eps, eps), n_seg0, None)])
        tups.extend([((-eps, eps), (eps, eps), n_seg1, None)])
        tups.extend([((eps, eps), (y, z), n_seg0, None)])
        tups.extend([((y, z), (y, 0), n_seg0, None)])
        tups.extend([((-y, 0), (-y, -z), n_seg0, None)])
        tups.extend([((-y, -z), (-eps, -eps), n_seg0, None)])
        tups.extend([((eps, -eps), (y, -z), n_seg0, None)])
        tups.extend([((y, -z), (y, 0), n_seg0, None)])
        tups.extend([((-eps, -eps), (eps, -eps), n_seg1, 1 + 0j)])

        new_tups = []
        for yoff, zoff in [(0, self.base)]:
            new_tups.extend(
                [
                    ((0, y0 + yoff, z0 + zoff), (0, y1 + yoff, z1 + zoff), ns, ev)
                    for ((y0, z0), (y1, z1), ns, ev) in tups
                ]
            )

        return new_tups
