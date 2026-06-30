"""A delta loop built top-down, with the feed terminal *discovered* by flying a
slant into a plane rather than computed.

The other drone-built variants fix the bottom feed gap and grow the triangle
upward (``delta_loop_marked`` flies each slant by its length; ``delta_loop_
reflected`` flies pen-up to read off the top corner, then reflects). This one
inverts that: it starts at the **top centre**, flies right to the top corner,
turns down onto the slant, and lets ``Drone.forward_to_plane`` fly until it
crosses the feed plane ``y = eps`` -- the plane ``eps`` from the loop's centre
line. Neither the slant length nor the feed's height is computed in the script;
both fall out of where the slant meets that plane. From there it borrows the
*last two* variants' techniques wholesale: the two right-side vertices become
**named pins** (``mark``/``line_to``) and their **reflections** across the
``y = 0`` plane (``ry``) give the left side, so the four-edge loop closes with
no trig at all in this module.

The loop is vertical, in the plane x = 0, fed by a short driven gap at the
bottom centre and opening upward:

        R-----------L      top, flown corner-to-corner from the centre start
         \\         /
          \\       /        two equal slants; the right one is flown to the
           \\     /         feed plane, the left is its reflection
            T~~~~S          short driven feed gap straddling y = 0 (T = ry(S))

Because the build is anchored at the *top* (param ``top``), the feed sits
wherever the slant lands -- a little below ``top`` -- rather than at a fixed
height; with the defaults it lands near 7 m, matching the sibling loops' base.
"""

from types import MappingProxyType

from ... import AntennaBuilder, Drone


class Builder(AntennaBuilder):
    default_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            # Height of the top edge -- the centre point the build starts from.
            # The feed height is NOT set here; it is whatever height the slant
            # has dropped to when it crosses the feed plane (see build_wires).
            "top": 10.0,
            # Scales the "go right" distance (half the top width), so it tunes
            # the loop size for resonance. The slant length follows from it and
            # the tilt angle via forward_to_plane.
            "length_factor": 1.0,
            # Tilt of each slant from vertical. 30 deg -> a 60 deg apex, the
            # classic near-equilateral delta loop. The downward turn at the top
            # corner is -(90 + angle_deg) deg (~ -120, i.e. a ~240/300 deg
            # swing) so the nose points down the slant toward the feed plane.
            "angle_deg": 30.0,
            "ui_params": MappingProxyType(
                {
                    "target_z0": 100.0,
                    "default_view": "yz",  # loop lies in the x = 0 plane
                    "length_factor": {"min": 0.85, "max": 1.15},
                    "angle_deg": {"min": 10.0, "max": 60.0},
                }
            ),
        }
    )

    def build_wires(self):
        eps = 0.05
        wavelength = 299.792458 / self.design_freq
        quarter = 0.25 * wavelength
        # "Go right" distance: half the top width, wl-scaled so length_factor
        # tunes the loop. The slant length and the feed height stay unknown to
        # this script -- forward_to_plane discovers where the slant crosses the
        # feed plane.
        half_top = (wavelength / 6.0) * self.length_factor

        n_seg0 = self.nominal_nsegs
        n_seg1 = max(3, self.nominal_nsegs // 7)

        def ry(p):
            return p[0], -p[1], p[2]

        drone = Drone(nominal_nsegs=self.nominal_nsegs, ref=quarter)

        # Fly the right half pen-up to FIND its two vertices, no trig:
        #   - start at the top centre, nose +y ("right"), up +x so every turn
        #     stays in the x = 0 plane;
        #   - fly half the top width to the top-right corner R;
        #   - turn down onto the slant and fly until we cross the feed plane
        #     y = eps (the plane eps from the centre line) to land on the
        #     feed terminal S. forward_to_plane solves the slant length for us.
        drone.move_to((0.0, 0.0, self.top))
        drone.face(heading=(0.0, 1.0, 0.0), up=(1.0, 0.0, 0.0))
        R = drone.forward(half_top).position
        drone.yaw(-(90.0 + self.angle_deg))
        S = drone.forward_to_plane((0.0, 1.0, 0.0, eps)).position

        # The left half is the right half reflected across the y = 0 plane.
        L, T = ry(R), ry(S)

        # Complete the loop with named pins (the marked variant's idiom): pin
        # all four corners, then stitch the perimeter S -> R -> L -> T with the
        # pen down (the drone works out each segment's direction), and finally
        # the short driven gap T -> S.
        drone.cut().move_to(S).mark("S")
        drone.move_to(R).mark("R")
        drone.move_to(L).mark("L")
        drone.move_to(T).mark("T")

        drone.move_to(S).pay_out()
        drone.line_to("R", nsegs=n_seg0)  # up the right slant
        drone.line_to("L", nsegs=n_seg0)  # across the top
        drone.line_to("T", nsegs=n_seg0)  # down the left slant
        drone.cut().move_to(T).feed(1 + 0j).line_to("S", nsegs=n_seg1)  # feed gap

        return drone.wires()
