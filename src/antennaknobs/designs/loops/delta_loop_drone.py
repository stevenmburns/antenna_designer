"""The same delta loop as ``delta_loop.py``, authored with the 3D-turtle
:class:`~antennaknobs.drone.Drone` instead of absolute corner coordinates.

It is a didactic twin: ``test_drone.py`` asserts this produces byte-identical
wires to ``delta_loop.Builder``. The point is to show that one antenna can be
expressed in different idioms — here, *describe the flight* rather than solve
for the apex coordinates.

The loop is a downward-pointing triangle fed at the bottom apex: a wide top
edge A-B at height ``base`` and two slanted sides meeting at the narrow feed
gap S-T just below. The geometry *sizing* (the apex height ``h`` that makes
the perimeter equal the driver length) is identical to delta_loop — the trig
is the physics. What changes is the *construction*: fly the right side, turn
the exterior angle, fly the top, turn, fly the left side, then close across
the feed gap back to the start.
"""

import math
from types import MappingProxyType

from ... import AntennaBuilder, Drone


class Builder(AntennaBuilder):
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
        quarter = 0.25 * wavelength
        driver = wavelength * self.length_factor

        theta = math.radians(self.angle_deg)
        cos_theta, tan_theta = math.cos(theta), math.tan(theta)

        # Apex height that makes the total wire perimeter equal the driver
        # length (same closed form delta_loop uses), plus the side and top
        # leg lengths that follow from it.
        h = (cos_theta * (driver - 2 * eps) + 2 * eps) / (2 * (cos_theta + 1))
        side = (h - eps) / cos_theta  # slanted side S->A (== B->T)
        top = 2 * h  # horizontal top A->B

        # Bottom-right feed terminal; the loop is planar in x = 0.
        S = (0.0, eps, b - (h - eps) * tan_theta)

        n_body = self.nominal_nsegs
        n_feed = max(3, self.nominal_nsegs // 7)

        drone = Drone(position=S, nominal_nsegs=n_body, ref=quarter)
        # Start nosed up-and-out toward the top-right corner A, banked so the
        # loop plane (x = 0) is the turning plane — then yaw stays in-plane.
        drone.face(heading=(0.0, cos_theta, math.sin(theta)), up=(1.0, 0.0, 0.0))

        drone.pay_out()
        drone.forward(side, nsegs=n_body)  # S -> A  (right side)
        drone.yaw(180 - self.angle_deg)  # exterior angle at A
        drone.forward(top, nsegs=n_body)  # A -> B  (top edge)
        drone.yaw(180 - self.angle_deg)  # exterior angle at B
        drone.forward(side, nsegs=n_body)  # B -> T  (left side)
        drone.feed(1 + 0j)
        drone.close(nsegs=n_feed)  # T -> S  (driven feed gap, fly home)

        return drone.wires()
