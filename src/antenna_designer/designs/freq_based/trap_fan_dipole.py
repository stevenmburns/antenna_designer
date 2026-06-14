"""Four-band trapped fan dipole — combines `fandipole` geometry with the
`trap_dipole` Load(parallel=True) idiom.

Two physical fan-dipole spokes (per arm), each broken by a parallel-LC
trap segment partway out. At the trap's resonant frequency, the tank is
high-Z and interrupts the spoke's current — only the inner stub radiates,
behaving as a shorter dipole. Well below the trap, the tank looks
inductive and electrically lengthens the spoke so the full physical
length sets the low-band tuning.

Default tuning:

  spoke 0 (long)   — full length resonant near 17m (18.16 MHz);
                     trap at 12m (24.97 MHz) shortens it to a 12m dipole.
  spoke 1 (short)  — full length resonant near 15m (21.38 MHz);
                     trap at 10m (28.47 MHz) shortens it to a 10m dipole.

→ four bands (10m / 12m / 15m / 17m) from a single shared feed.

The cone geometry is the same fan layout as `fandipole.py`: each spoke
shares a common slope direction (0, Zc, −Zs) from the cone apex outward.
Each arm of each spoke is broken into

    S → A[i]              (cone segment)
    A[i] → trap_inner     (inner outer segment)
    trap_inner → trap_outer  (one segment, named "trap_<sign>_b<i>")
    trap_outer → tip      (outer segment)

with two arms per spoke (mirrored via ry()), so a 2-spoke design has
four traps total. The trap segment is wire-interior, so its basis
function carries the actual arm current and Load(parallel=True) acts
exactly as NEC2's ld_card type-1 would.
"""

from ... import AntennaBuilder
from ...network import Driven, Load, Network, PortAtEdge

import math
from types import MappingProxyType


C_LIGHT_MHZ_M = 299.792458


def _resonant_C_pF(L_uH: float, freq_mhz: float) -> float:
    """C in pF such that ω² LC = 1 at `freq_mhz`."""
    omega = 2 * math.pi * freq_mhz * 1e6
    return 1.0 / (omega**2 * L_uH * 1e-6) * 1e12


# Defaults: 5 µH traps, C chosen so each trap LC-resonates at its trap_freq.
# length_factor ≈ 0.49 captures the usual end-effect shortening on a fan
# spoke; tweak per band if a sweep shows the resonance off-target.
# All four bands tuned by `cli optimize --resonance`. Both traps have
# their L/C decoupled from the usual L↔C lockstep at ω₀ = trap_freq, in
# opposite directions:
#   - Band 1: L pushed DOWN to ~0.2 µH (with C → ~159 pF). The 10m trap
#     is only ~7 MHz above 15m, so the stock 5 µH puts ~+j1.4 kΩ on the
#     spoke at 15m — enough to smother every series resonance. Cutting L
#     weakens that loading; the 15m spoke can then find resonance at its
#     physical length, while the tank still LC-resonates at 28.47 to
#     interrupt the spoke at 10m.
#   - Band 0: L pushed UP to ~11.85 µH (with C → ~3.4 pF). Heavier
#     loading at 17m concentrates current toward the feed and lowers the
#     resonant Re(Z) toward 50 Ω, shaving max SWR50 from 1.31 to 1.27.
#     Beyond ~11.9 µH the 17m series resonance merges with the adjacent
#     parallel-resonance pole and disappears.
# Slope=0.62 (instead of the original 0.5) drops the resistances
# uniformly so they straddle 50 Ω: 17m and 10m sit at the extremes with
# matching SWR50≈1.21, and the in-band bands (15m/12m) come in lower.
# A small `trap1_freq_shift` = 0.98 nudges 10m into a different mode
# where Re(Z) ≈ 50 Ω (the trap doesn't fully open at 10m, the outer
# extension joins in, and the inner settles at ~0.40·λ/2 instead of
# ~0.50). 10m's SWR50 drops from 1.20 to 1.02 with the other three
# bands largely unchanged. (Band-0 shift has near-zero leverage on
# Re17 — kept at 1.0.)
# Length factors tuned against PysimEngine with BSplinePySim(degree=2)
# at nominal_nsegs=41 (initial pass at N=21, final refinement at N=41).
# Bs2 is the only basis whose Z(N) sequence stays flat as N grows on the
# heavily-loaded 17m and 10m bands — PyNEC, triangular, and sinusoidal
# all drift by 20+ Ω from N=21 to N=81 there. Tuning against Bs2 gives
# defaults that survive engine convergence rather than tracking a
# specific N's discretization artefact.
# Final SWR50 at target freqs (Bs2 @ N=41): 17m=1.04, 15m=1.02, 12m=1.19, 10m=1.13.
# Same antenna under PyNEC@21 gives larger SWRs (notably ~1.29 at 17m) —
# the engines disagree on the trap-loaded bands by more than any tuning
# tweak can absorb. Plan to verify by measurement after build.
_BAND_17_12 = {
    "full_freq": 18.1575,
    "trap_freq": 24.97,
    "full_length_factor": 0.442400,
    "inner_length_factor": 0.498100,
    # L=3 µH well below the parallel-resonance "cliff" (the regime
    # where the 17m series resonance merges with the adjacent parallel-
    # resonance pole at L ≳ 11.9 µH). The high-L cliff gives a marginally
    # better ideal SWR50 (1.05 vs 1.07) but the resonance Q is so high
    # there that ±1 cm length error sends SWR to 3-5. At L=3 the ±1 cm
    # tolerance is ≤ 1.17 SWR — buildable. C tracks at LC-resonance for
    # 24.97 MHz given this L.
    "trap_L_uH": 3.0,
    "trap_C_pF": _resonant_C_pF(L_uH=3.0, freq_mhz=24.97),
}

_BAND_15_10 = {
    "full_freq": 21.383,
    "trap_freq": 28.47,
    "full_length_factor": 0.453200,
    "inner_length_factor": 0.447400,
    # L=1 µH — practical minimum for a hand-wound air-core coil
    # (sub-µH coils have inductance dominated by stray/lead effects and
    # are hard to build reproducibly). C tracks at LC-resonance for
    # 28.47 MHz. The trap1_freq_shift below pulls effective ω₀ slightly
    # to bring 10m into a clean mode-jump resonance.
    "trap_L_uH": 1.0,
    "trap_C_pF": _resonant_C_pF(L_uH=1.0, freq_mhz=28.47),
}


class Builder(AntennaBuilder):
    default_params = MappingProxyType(
        {
            # `freq` is the operating frequency the sweep UI defaults to;
            # nothing in the geometry reads it directly. `design_freq` is
            # read by the PyNEC engine when synthesising virtual-port
            # stubs and as a sweep-range anchor — set it to the middle of
            # the band span so per-engine defaults make sense.
            "design_freq": 21.0,
            "freq": 28.47,
            "base": 7.0,
            # Same fan-spoke slope as fandipole — each spoke drops at this
            # slope from the cone apex outward (y_dir=Zc, z_dir=−Zs).
            # Slope of the inverted-vee arms (descent angle from horizontal).
            # Steeper slope lowers radiation resistance for all four bands
            # simultaneously. Tuned to ~0.62 so the resonant Re(Z) values
            # straddle 50 Ω evenly — 17m and 10m end up at the two extremes
            # (~60 Ω and ~42 Ω) with matching SWR50 ≈ 1.21, the lower bound
            # for max-SWR given the geometry.
            "slope": 0.68,
            # Length (m) of the single-segment "trap wire" carrying the
            # named Load port. Should be much shorter than λ so radiation
            # from the trap segment itself is negligible.
            "trap_seg_m": 0.05,
            # Per-trap resonant-frequency multipliers (dimensionless).
            # Each shift scales the trap's effective ω₀ by its value;
            # internally this is applied as C_eff = trap_C_pF / shift²
            # (L held fixed). Useful as a fine knob when the high-band
            # resonance lands slightly off after a length retune — moving
            # the trap a few percent shifts the inner-stub electrical
            # length without touching the geometry. Default 1.0 = no shift.
            "trap0_freq_shift": 1.0,
            # Set slightly below 1.0 so band-1's trap is just past resonance
            # at 10m. The trap doesn't fully open, the outer extension joins
            # in, and the inner sits in a different mode (~0.40·λ/2 instead
            # of ~0.50). Net effect: 10m's resonant Re(Z) jumps from ~42 Ω
            # toward ~50 Ω, SWR50 at 10m drops from 1.20 to 1.02. (Band 0's
            # shift has almost no leverage on Re17 — left at 1.0.)
            "trap1_freq_shift": 0.95,
            # Pinned at 2 — the design is hard-coded to two parallel
            # spokes. Exposed in default_params (with min=max=2 in
            # ui_params below) so the bands group has a `repeat_count`
            # to reference, satisfying the schema adapter's contract.
            "n_bands": 2,
            # UI exposure for the `bands` group + per-leaf range hints.
            # tuple-of-dicts in default_params becomes a ParamGroupSpec
            # in the schema; the `bands` dict below tells the adapter how
            # to render each band-row in the sidebar.
            "ui_params": MappingProxyType(
                {
                    "n_bands": {"min": 1, "max": 2, "step": 1},
                    "bands": {
                        "label_template": "band {i}",
                        "repeat_count": "n_bands",
                        "max_repeats": 2,
                        "link_meas_freq_to_param": "full_freq",
                        "full_freq": {
                            "min": 13.5,
                            "max": 30.2,
                            "step": 0.001,
                            "precision": 3,
                            "unit": " MHz",
                        },
                        "trap_freq": {
                            "min": 13.5,
                            "max": 30.2,
                            "step": 0.001,
                            "precision": 3,
                            "unit": " MHz",
                        },
                        "full_length_factor": {
                            "min": 0.30,
                            "max": 0.60,
                            "step": 0.0005,
                            "precision": 4,
                        },
                        "inner_length_factor": {
                            "min": 0.30,
                            "max": 0.60,
                            "step": 0.0005,
                            "precision": 4,
                        },
                        "trap_L_uH": {
                            "min": 0.1,
                            "max": 20.0,
                            "step": 0.01,
                            "precision": 3,
                            "unit": " µH",
                        },
                        "trap_C_pF": {
                            "min": 0.5,
                            "max": 500.0,
                            "step": 0.01,
                            "precision": 3,
                            "unit": " pF",
                        },
                    },
                }
            ),
            # Per-band tuning. Each entry: low-band freq (full antenna),
            # high-band freq (trap resonance + inner stub length), per-band
            # length-factor knobs, and the trap's L/C.
            "bands": (_BAND_17_12, _BAND_15_10),
        }
    )

    # Per-band tuning variants. Each one anchors `freq` at the band's
    # target so `cli optimize --resonance --params bands.<i>.<which>_length_factor`
    # drives the right resonance into place. Used to refine defaults; once
    # the optimized length factors are folded back into the `bands` tuple
    # above these are just convenience entry points.
    band0_inner_params = MappingProxyType({**default_params, "freq": 24.97})
    band1_inner_params = MappingProxyType({**default_params, "freq": 28.47})
    band0_full_params = MappingProxyType({**default_params, "freq": 18.1575})
    band1_full_params = MappingProxyType({**default_params, "freq": 21.383})

    def build_wires(self):
        eps = 0.01
        radius = 0.12

        n_bands = int(self.n_bands)
        if n_bands != 2:
            raise ValueError(
                f"trap_fan_dipole is a 2-spoke design; got n_bands={n_bands}"
            )
        bands = tuple(self.bands)[:n_bands]

        Zc = 1 / math.sqrt(1 + self.slope**2)
        Zs = self.slope * Zc

        def ry(p):
            return p[0], -p[1], p[2]

        S = (0, eps, 0)
        T = ry(S)

        # Each band's spoke originates from a short horizontal pigtail at
        # ±radius in x from the feed. The pigtail puts band 0 and band 1
        # on opposite sides of the feed (separated by 2·radius along x);
        # beyond the pigtail each spoke extends in the shared inverted-vee
        # direction (0, Zc, −Zs). The two spokes are then parallel
        # everywhere except the small horizontal segment near the feed,
        # which keeps the `slope` parameter from also controlling the
        # band-spacing direction (a problem the original cone-apex layout
        # had — slope and band spacing were entangled near the feed).
        A = [
            (+radius, S[1], S[2]),  # band 0 on the +x side
            (-radius, S[1], S[2]),  # band 1 on the −x side
        ]

        def dist(p0, p1):
            return math.sqrt(sum((x0 - x1) ** 2 for x0, x1 in zip(p0, p1)))

        # Outward direction shared by every spoke beyond the cone (the
        # "fan" direction): pure (0, Zc, −Zs).
        def offset_outward(p, q):
            return (p[0], p[1] + q * Zc, p[2] - q * Zs)

        trap_seg = float(self.trap_seg_m)

        # For each spoke, derive the three outer waypoints along the
        # shared outward direction:
        #   trap_inner = A + q1
        #   trap_outer = A + q1 + trap_seg
        #   tip        = A + q1 + trap_seg + q2
        # with q1 chosen so dist(S,A) + q1 = inner half-length, and the
        # full half-length (q1 + trap_seg + q2) set by the low-band freq.
        spokes = []
        for i, b in enumerate(bands):
            full_half = float(b["full_length_factor"]) * (
                0.5 * C_LIGHT_MHZ_M / float(b["full_freq"])
            )
            inner_half = float(b["inner_length_factor"]) * (
                0.5 * C_LIGHT_MHZ_M / float(b["trap_freq"])
            )

            q1 = inner_half - dist(S, A[i])
            q2 = full_half - inner_half - trap_seg
            if q1 <= 0:
                raise ValueError(
                    f"band {i}: inner half-length {inner_half:.3f} m is "
                    f"shorter than the cone segment {dist(S, A[i]):.3f} m — "
                    "shorten the cone or pick a lower trap_freq"
                )
            if q2 <= 0:
                raise ValueError(
                    f"band {i}: full half-length {full_half:.3f} m leaves no "
                    f"room past the trap (inner {inner_half:.3f} + trap "
                    f"{trap_seg:.3f}) — pick a lower full_freq or shorter inner"
                )

            trap_in = offset_outward(A[i], q1)
            trap_out = offset_outward(A[i], q1 + trap_seg)
            tip = offset_outward(A[i], q1 + trap_seg + q2)
            spokes.append((trap_in, trap_out, tip))

        n_seg0 = self.nominal_nsegs
        # Feed-wire segment count: see fandipole.py for the reason — pysim
        # triangular basis needs at least one interior knot, n_seg=1 trips
        # an argmin-of-empty in triangular._feed_basis_indices.
        n_seg1 = max(3, self.nominal_nsegs // 7)
        n_outer = max(5, self.nominal_nsegs // 2)

        tups = []
        for i, (trap_in, trap_out, tip) in enumerate(spokes):
            # +y arm
            tups.append((S, A[i], n_seg0, None))
            tups.append((A[i], trap_in, n_seg0, None))
            tups.append((trap_in, trap_out, 1, None, f"trap_p_b{i}"))
            tups.append((trap_out, tip, n_outer, None))

            # −y arm (mirror via ry)
            Ay = ry(A[i])
            tin_y = ry(trap_in)
            tout_y = ry(trap_out)
            tip_y = ry(tip)
            tups.append((T, Ay, n_seg0, None))
            tups.append((Ay, tin_y, n_seg0, None))
            tups.append((tin_y, tout_y, 1, None, f"trap_n_b{i}"))
            tups.append((tout_y, tip_y, n_outer, None))

        # Feed wire — named "feed", source supplied by build_network().
        tups.append((T, S, n_seg1, None, "feed"))

        # Lift to base height.
        zoff = self.base
        lifted = []
        for t in tups:
            (x0, y0, z0), (x1, y1, z1) = t[0], t[1]
            lifted.append(((x0, y0, z0 + zoff), (x1, y1, z1 + zoff), *t[2:]))
        return lifted

    def build_network(self):
        bands = tuple(self.bands)
        shifts = (float(self.trap0_freq_shift), float(self.trap1_freq_shift))
        ports = {"feed": PortAtEdge("feed")}
        branches = []
        for i, b in enumerate(bands):
            L = float(b["trap_L_uH"]) * 1e-6
            # Shift the tank's LC resonance by `shifts[i]` (dimensionless).
            # ω₀ = 1/√(LC), so scaling ω₀ by s with L fixed scales C by 1/s².
            C = float(b["trap_C_pF"]) * 1e-12 / shifts[i] ** 2
            for sign in ("p", "n"):
                name = f"trap_{sign}_b{i}"
                ports[name] = PortAtEdge(name)
                branches.append(Load(port=name, l=L, c=C, parallel=True))
        return Network(
            ports=ports,
            branches=branches,
            sources=[Driven(port="feed", voltage=1 + 0j)],
        )
