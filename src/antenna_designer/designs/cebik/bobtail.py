"""Bobtail curtain: a 3-element vertically-polarised broadside array
(L. B. Cebik, W4RNL).

The bobtail is the electrical big brother of the half-square: THREE roughly
1/4-wavelength vertical radiators, spaced about 1/2 wavelength apart, joined
along the top by a continuous ~1-wavelength phasing wire. All three verticals
end up driven in phase, so the array is VERTICALLY POLARISED and fires
bidirectionally broadside to the plane of the wires, with a tighter pattern
and a few dB more gain than the half-square (~5.1 dBi, max at a low ~19 deg
elevation over ground).

Only the centre vertical is physically fed -- at its base, the classic
"matching tank at the bottom of the centre wire" point. This is a high,
reactive impedance (hundreds of ohms and up), which is why the real antenna
uses a parallel-tuned tank rather than a direct coax feed. The two OUTER
verticals are passive, open at the bottom, and excited entirely through the
top phasing wire.

Cebik's 40 m proportions: verticals ~0.243 wl, half-span (centre-to-outer)
~0.541 wl, so the full top wire is ~1.083 wl.

Geometry, in the framework's (x, y, z) convention:
  - y : the long axis (the three verticals sit at y = -span, 0, +span)
  - z : height; leg bottoms at `base`, top wire at `base + vert`
  - x : firing axis; radiation is broadside off +/- x
The structure is planar in x = 0.

    C1=========C2=========C3    z = base + vert   (top wire, ~1.08 wl)
    |          |          |
    |          |          |     three verticals, ~0.243 wl
    |          F          |
    A1         A2         A3     z = base   (outer ends open; centre base-fed)
"""

from ... import AntennaBuilder
from types import MappingProxyType


class Builder(AntennaBuilder):
    default_params = MappingProxyType(
        {
            "design_freq": 28.57,
            "freq": 28.57,
            # Height of the open leg ends above ground.
            "base": 3.0,
            # Vertical-radiator length as a fraction of a wavelength
            # (Cebik's 40 m model: ~0.243 wl).
            "vert_frac": 0.243,
            # Half-span: centre-to-outer horizontal spacing as a fraction of
            # a wavelength (~0.541 wl) -> full top wire ~1.083 wl.
            "span_frac": 0.541,
            # Overall scale knob. Unlike the half-square, the base feed is
            # inherently high-Z and reactive (tank-matched), so this is not
            # tuned for resonance -- length_factor ~1.0 is Cebik's max-gain
            # proportion; peak gain/pattern, not X->0, is the design target.
            "length_factor": 1.0,
            "ui_params": MappingProxyType(
                {
                    # High, reactive feed (tank-matched in practice); reference
                    # SWR to a representative open-wire/tank impedance.
                    "target_z0": 300.0,
                    "default_view": "yz",
                    "length_factor": {
                        "min": 0.9,
                        "max": 1.1,
                        "step": 0.001,
                        "precision": 4,
                    },
                }
            ),
        }
    )

    def build_wires(self):
        eps = 0.05

        wavelength = 299.792458 / self.design_freq
        quarter = 0.25 * wavelength

        vert = self.vert_frac * wavelength * self.length_factor
        span = self.span_frac * wavelength * self.length_factor

        z_bot = self.base
        z_top = self.base + vert

        def nsegs(length):
            n = max(3, round(self.nominal_nsegs * length / quarter))
            return n if n % 2 == 1 else n + 1

        tups = []

        # Top phasing wire: -span -> 0 -> +span, split at the centre so the
        # centre vertical shares the junction node.
        tups.append(((0.0, -span, z_top), (0.0, 0.0, z_top), nsegs(span), None))
        tups.append(((0.0, 0.0, z_top), (0.0, span, z_top), nsegs(span), None))

        # Outer verticals (passive, open at the bottom).
        tups.append(((0.0, -span, z_top), (0.0, -span, z_bot), nsegs(vert), None))
        tups.append(((0.0, span, z_top), (0.0, span, z_bot), nsegs(vert), None))

        # Centre vertical: passive from the top down to just above the base,
        # then a one-segment driven gap at the base (the matching-tank point).
        feed = 2 * eps
        tups.append(
            ((0.0, 0.0, z_top), (0.0, 0.0, z_bot + feed), nsegs(vert - feed), None)
        )
        tups.append(((0.0, 0.0, z_bot + feed), (0.0, 0.0, z_bot), 1, 1 + 0j))

        return tups
