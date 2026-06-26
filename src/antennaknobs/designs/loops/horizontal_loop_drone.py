"""A horizontal square loop, vertex-fed, authored with the 3D-turtle Drone.

The simplest drone example there is: a flat square loop built by flying four
equal sides with a 90 degrees turn at each corner -- no trig at all. Contrast
``delta_loop_drone``, where the side and top lengths still need the apex-height
formula; here every side is the same length and the corners are just right
angles.

The loop lies flat in the plane ``z = base``. Each side is a quarter wavelength
scaled by a common ``length_factor`` (so the perimeter is ~1 wavelength at
``length_factor = 1`` -- a full-wave horizontal loop). It is driven by a short
one-segment gap right at one corner: a *vertex feed* (unlike ``horizontal_loop``,
which feeds the midpoint of a side).

        D-----------C       (loop lies flat at z = base, viewed from above)
        |           |
        |           |
        |           |
        A==>--------B       feed gap at vertex A, along side A->B
"""

from types import MappingProxyType

from ... import AntennaBuilder, Drone


class Builder(AntennaBuilder):
    default_params = MappingProxyType(
        {
            "design_freq": 28.57,
            "freq": 28.57,
            # Height of the flat loop plane above ground.
            "base": 5.0,
            # Each side is a quarter wavelength times this; length_factor = 1
            # gives a ~1 wl perimeter (full-wave loop). The optimiser trims it
            # for resonance.
            "length_factor": 1.0,
            "ui_params": MappingProxyType(
                {
                    "target_z0": 100.0,
                    "default_view": "xy",
                    "length_factor": {"min": 0.9, "max": 1.1},
                }
            ),
        }
    )

    def build_wires(self):
        eps = 0.05
        wavelength = 299.792458 / self.design_freq
        quarter = 0.25 * wavelength

        side = quarter * self.length_factor  # each side ~ a quarter wave
        h = side / 2.0
        feed = 2 * eps  # length of the one-segment driven gap

        # Start at the feed vertex A = (-h, -h, base). The drone's default pose
        # faces +x with 'up' = +z, so the loop is horizontal and yaw(90) turns
        # stay in the z = base plane -- the whole figure needs no trig.
        drone = Drone(
            position=(-h, -h, self.base),
            nominal_nsegs=self.nominal_nsegs,
            ref=quarter,
        )

        drone.feed(1 + 0j).forward(feed, nsegs=1)  # driven gap at vertex A
        drone.pay_out()
        drone.forward(side - feed)  # finish side A->B
        drone.yaw(90).forward(side)  # side B->C
        drone.yaw(90).forward(side)  # side C->D
        drone.yaw(90).close()  # side D->A, fly home (exact close)

        return drone.wires()
