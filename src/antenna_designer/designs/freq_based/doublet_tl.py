"""Non-resonant doublet fed by an open-wire line (modelled as a TL) to a tuner.

A doublet is a centre-fed flat-top whose length is deliberately *not* λ/2 on
the operating band, so its feedpoint impedance is high and reactive; in
practice it is fed with open-wire / ladder line down to an antenna tuner
that resolves the match. Here the feeder is one `network.TL` (a single-ended
lossless line — the element that correctly feeds a collinear centre-fed
dipole), running from a virtual driver (the tuner output) up to the dipole's
centre feed edge.

Topology (framework x, y, z convention):
  - y : the doublet's long axis; one continuous flat-top, centre-fed
  - z : height (`base`); the wire is horizontal
  - x : unused

The driving-point impedance is read at the bottom of the feeder — **what the
tuner has to match.** As with `sterba`, the tuner itself is not modelled as a
circuit; SWR is referenced to the feeder impedance (`ui_params["target_z0"]`),
i.e. the open-wire-line SWR (constant along a lossless line) the tuner then
absorbs. Vary `feeder_factor` to watch the feedpoint impedance rotate along
the line (a quarter-wave feeder inverts it).

length_factor is the total doublet length in wavelengths (default 1.28λ, the
classic Extended-Double-Zepp non-resonant length, ~100−j600 Ω at the
feedpoint). feeder_factor is the feeder's electrical length in wavelengths,
kept inside (0, ½) to stay off the ideal-line kλ/2 singularity. z0 is the
feeder (open-wire line) impedance.

This is the all-validated-elements cousin of the DiffTL experiments: a single
TL feeds the collinear dipole's through-current correctly, where an ideal
4-terminal DiffTL cannot (the dipole radiates via the common/through mode, not
the differential mode a DiffTL drives). Driven via the network spec on
PysimEngine; the TL→tuner virtual-driver translation is the same one
`delta_looparray_network` uses. Compare with `doublet_2wire`, the real-wire
feeder version.
"""

from ... import AntennaBuilder
from ...network import Driven, Network, PortAtEdge, PortVirtual, TL

from types import MappingProxyType


class Builder(AntennaBuilder):
    default_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "base": 10.0,
            # Total doublet length in wavelengths. 1.28λ is the classic
            # Extended Double Zepp — deliberately non-resonant, so the
            # feedpoint is high and reactive and genuinely wants a tuner.
            "length_factor": 1.28,
            # Feeder electrical length in wavelengths. Kept in (0, 0.5) so
            # the ideal TL stays off its kλ/2 admittance singularity.
            "feeder_factor": 0.3,
            "z0": 600.0,  # open-wire / ladder-line impedance
            "ui_params": MappingProxyType(
                {
                    # SWR referenced to the feeder impedance: the open-wire-
                    # line SWR (constant along a lossless line) that the tuner
                    # at the bottom then matches out.
                    "target_z0": 600.0,
                    "sweep_policy": {"band_locked": True},
                    "length_factor": {
                        "min": 0.4,
                        "max": 2.0,
                        "step": 0.001,
                        "precision": 4,
                    },
                    # Stop short of 0.5λ: at exactly kλ/2 the ideal line is
                    # singular (sin βl = 0).
                    "feeder_factor": {
                        "min": 0.05,
                        "max": 0.45,
                        "step": 0.005,
                        "precision": 3,
                    },
                    "z0": {"min": 200.0, "max": 600.0, "step": 5.0},
                }
            ),
        }
    )

    def build_wires(self):
        wavelength = 299.792458 / self.design_freq
        half = 0.5 * self.length_factor * wavelength
        z = self.base
        eps = 0.05  # half-width of the centre feed edge

        n_seg0 = self.nominal_nsegs
        n_seg1 = max(3, n_seg0 // 7)

        # One continuous flat-top along y, centre-fed: two arms plus a short
        # bridging feed edge spanning the gap (the standard delta-gap that
        # drives the dipole's through-current). The TL attaches at "feed".
        return [
            ((0.0, eps, z), (0.0, half, z), n_seg0, None, None),
            ((0.0, -half, z), (0.0, -eps, z), n_seg0, None, None),
            ((0.0, -eps, z), (0.0, eps, z), n_seg1, None, "feed"),
        ]

    def build_network(self):
        wavelength = 299.792458 / self.design_freq
        feeder_len = self.feeder_factor * wavelength
        return Network(
            ports={
                "feed": PortAtEdge("feed"),
                "tuner": PortVirtual("tuner"),
            },
            branches=[
                # Single-ended open-wire feeder from the tuner output up to the
                # dipole centre. The TL transforms the (high, reactive)
                # feedpoint impedance down the line to whatever the tuner sees.
                TL(a="tuner", b="feed", z0=self.z0, length=feeder_len),
            ],
            sources=[Driven(port="tuner", voltage=1 + 0j)],
        )
