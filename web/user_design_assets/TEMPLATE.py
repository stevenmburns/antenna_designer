"""A complete, working example antenna. Copy it and make it your own.

HOW TO USE THIS FILE
--------------------
1. Copy it to a new file in this same folder, e.g. ``my_dipole.py``. The
   file name (lowercase, words joined by underscores) becomes the antenna's
   name in the app: ``my_dipole.py`` shows up as "user.my_dipole".
2. Change the numbers in ``default_params`` and the geometry in
   ``build_wires``.
3. Refresh the web page. Your antenna appears under "Your designs". If you
   made a mistake, the page shows the error so you can fix it.

NOT A PYTHON PROGRAMMER?
------------------------
Open Claude Code in this folder and just ask, for example:
    "make me a 40-meter off-center-fed dipole"
The CLAUDE.md file next to this one tells Claude everything it needs to
write a valid design and check its work.

THE RULES (keep it this simple)
-------------------------------
- One file = one antenna. The file name is the antenna name
  (lowercase_with_underscores.py, no spaces, no dots except ``.py``).
- Define a class named exactly ``Builder`` that subclasses ``AntennaBuilder``.
- Stay self-contained: only import from ``antenna_designer`` and the Python
  standard library. Don't import other antenna files.
"""

from types import MappingProxyType

from antenna_designer import AntennaBuilder


class Builder(AntennaBuilder):
    # Optional friendly name shown in the UI. Without it, the file name is used.
    label = "Example dipole"

    # The knobs. Every entry here becomes a slider in the UI. ``freq`` (the
    # measurement frequency in MHz) is special and should always be present.
    default_params = MappingProxyType(
        {
            "freq": 28.5,  # MHz -- where you measure SWR / impedance
            "half_length": 2.5,  # metres -- length of each arm (~1/4 wave on 10 m)
            "height": 10.0,  # metres -- height above ground
            # Optional UI hints. "default_view" sets the first 2-D view:
            # "xy" (top-down), "xz" or "yz" (from the side).
            "ui_params": MappingProxyType({"default_view": "xz"}),
        }
    )

    def build_wires(self):
        """Return the antenna as a list of straight wire segments.

        Each entry is ``(start, end, n_segments, feed)``:
          * ``start`` / ``end`` are ``(x, y, z)`` points in metres,
          * ``n_segments`` is how finely to divide that wire (more = finer,
            slower) -- ``self.nominal_nsegs`` is a sensible default,
          * ``feed`` is ``1 + 0j`` on the ONE segment driven by the
            transmitter, and ``None`` on every other wire.
        """
        h = self.half_length
        z = self.height
        eps = 0.01  # tiny half-gap at the centre where the feed point sits

        arm_segs = self.nominal_nsegs
        feed_segs = max(1, self.nominal_nsegs // 7)

        # A center-fed dipole lying along the y axis at height z:
        #
        #   tip(-h) ---- arm ---- (-eps)[ feed ](+eps) ---- arm ---- tip(+h)
        #
        left_tip = (0.0, -h, z)
        right_tip = (0.0, h, z)
        feed_lo = (0.0, -eps, z)
        feed_hi = (0.0, eps, z)

        return [
            (left_tip, feed_lo, arm_segs, None),  # left arm
            (feed_hi, right_tip, arm_segs, None),  # right arm
            (feed_lo, feed_hi, feed_segs, 1 + 0j),  # the driven feed gap
        ]
