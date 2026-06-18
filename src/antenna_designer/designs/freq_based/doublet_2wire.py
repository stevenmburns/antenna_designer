"""Non-resonant doublet fed by a *real* two-wire open-wire line to a tuner.

The geometric counterpart to `doublet_tl`: instead of modelling the open-wire
feeder as an ideal `network.TL`, the two feeder conductors are real parallel
wires with geometry. They drop from the dipole's centre down to a delta-gap
feed at the bottom (the tuner output), so the feeder carries a genuine
transmission-line current *and* radiates / couples in the near field — the
effects an ideal line omits.

Topology (framework x, y, z convention):
  - y : the doublet's long axis (the flat-top) and the feeder spacing
  - z : height; flat-top at `base`, feeder drops to `base - feeder_factor·λ`
  - x : unused

Current path is one continuous loop: +y tip → +y arm → down the +y feeder
wire → feed edge (driven) → up the −y feeder wire → −y arm → −y tip. The feed
edge at the bottom is the tuner output; its impedance is what the tuner must
match. SWR is referenced to the line impedance (`ui_params["target_z0"]`).

`spacing` (default ≈ 7.4 cm with the 0.5 mm default wire radius) sets the
line's characteristic impedance to ≈ 600 Ω, matching `doublet_tl`'s default
`z0` so the two can be compared directly. length_factor is the total doublet
length in wavelengths (default 1.28λ EDZ); feeder_factor is the feeder's
physical length in wavelengths (≈ electrical length for air-spaced line).

Compare against `doublet_tl`: the impedances and patterns agree to the extent
that the ideal line is a good model — differences are exactly the feeder's
radiation and near-field coupling. Runs on either engine.
"""

from ... import AntennaBuilder

from types import MappingProxyType


class Builder(AntennaBuilder):
    default_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "base": 10.0,
            "length_factor": 1.28,  # total doublet length in λ (EDZ)
            "feeder_factor": 0.3,  # feeder physical length in λ
            # Conductor spacing -> ~600 Ω with the 0.5 mm default wire radius
            # (z0 = 120·acosh(spacing / 2·radius)); matches doublet_tl's z0.
            "spacing": 0.074,
            "ui_params": MappingProxyType(
                {
                    "target_z0": 600.0,
                    "sweep_policy": {"band_locked": True},
                    "length_factor": {
                        "min": 0.4,
                        "max": 2.0,
                        "step": 0.001,
                        "precision": 4,
                    },
                    "feeder_factor": {
                        "min": 0.05,
                        "max": 0.6,
                        "step": 0.005,
                        "precision": 3,
                    },
                    "spacing": {
                        "min": 0.02,
                        "max": 0.30,
                        "step": 0.002,
                        "precision": 3,
                    },
                }
            ),
        }
    )

    def build_wires(self):
        wavelength = 299.792458 / self.design_freq
        half = 0.5 * self.length_factor * wavelength
        z = self.base
        sp = 0.5 * self.spacing  # half conductor spacing in y
        h = self.feeder_factor * wavelength  # feeder drop length
        zb = z - h  # feeder bottom (tuner output)

        n0 = self.nominal_nsegs
        n1 = max(3, n0 // 7)
        n_feed = max(3, round(n0 * h / (0.5 * wavelength)))

        # Continuous loop: arms at the top, two feeder wires dropping to a
        # delta-gap feed at the bottom. Endpoints are shared so the translator
        # chains them into one structure.
        return [
            ((0.0, sp, z), (0.0, half, z), n0, None),  # +y arm
            ((0.0, -half, z), (0.0, -sp, z), n0, None),  # -y arm
            ((0.0, sp, z), (0.0, sp, zb), n_feed, None),  # +y feeder wire
            ((0.0, -sp, zb), (0.0, -sp, z), n_feed, None),  # -y feeder wire
            ((0.0, -sp, zb), (0.0, sp, zb), n1, 1 + 0j),  # feed edge (tuner)
        ]
