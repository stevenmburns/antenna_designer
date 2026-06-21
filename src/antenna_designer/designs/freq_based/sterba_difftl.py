"""Sterba curtain, differential-transmission-line variant.

A research/demonstration variant of `freq_based.sterba` that replaces each
vertical *twisted-pair* phasing section with a single 4-terminal
differential transmission line (`network.DiffTL`) — the element NEC2's
`tl_card` cannot express, so this design is **PysimEngine-only**.

It reuses the validated all-wires `sterba` geometry, then surgically:
  - removes the 8 interior vertical riser wires (the A- and B-conductor
    risers at each of the n_cells+1 junctions),
  - names the four horizontal-section ends that meet at each junction
    (A-top, B-top, A-bot, B-bot), and
  - joins them with one `DiffTL` per junction whose differential mode
    carries the phasing current and whose common mode (`z0_cm`) carries
    the through-current.

What it is and isn't: it is the engine capability made concrete — it
recovers the curtain's *phasing* (the differential mode / twist) and most
of its *continuity* (the common mode), reaching ~7 dBi broadside. It is
NOT a better antenna than the all-wires `sterba` (~10.5 dBi): an ideal
line has no geometry, so it omits the verticals' near-field coupling to
the radiators — the last ~3.5 dB genuinely needs real conductors. Use
`freq_based.sterba` for a real antenna; use this to exercise/inspect the
differential-line element.

z0 is the differential-mode impedance (defaulted near the best-gain value,
not the physical ~526 ohm), z0_cm the common-mode impedance. Each DiffTL's
electrical length is `h * tl_length_factor`, where h is the half-wave riser
height (kept off exactly lambda/2 via length_factor=0.99 to dodge the ideal-
line singularity); tl_length_factor (default 1.0) phase-trims the lines
independently of the wire geometry.
"""

from ... import AntennaBuilder
from ...network import DiffTL, Driven, Network, PortAtEdge
from . import sterba

from types import MappingProxyType


class Builder(AntennaBuilder):
    default_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "base": 7.0,
            # Default off exactly 1.0 so the half-wave DiffTLs miss the
            # nodal-admittance singularity at length = lambda/2. The slider
            # range now spans 1.0; the singularity is at the *product*
            # length_factor * tl_length_factor == 1.0, so when running
            # length_factor at 1.0 keep tl_length_factor off 1.0 (or vice
            # versa) to stay clear of it.
            "length_factor": 0.99,
            "n_cells": 3,  # odd; interior DiffTLs = n_cells + 1
            "spacing": 0.04,
            "z0": 150.0,  # differential-mode impedance (near best gain)
            "z0_cm": 80.0,  # common-mode impedance
            # Multiplies the riser height h to set each DiffTL's electrical
            # length, decoupled from the physical geometry. 1.0 -> length = h
            # (the half-wave riser); tweak to phase-trim the lines without
            # moving any wires.
            "tl_length_factor": 1.0,
            "ui_params": MappingProxyType(
                {
                    "target_z0": 75.0,  # ~70 ohm driving point (measured)
                    "sweep_policy": {"band_locked": True},
                    "length_factor": {
                        "min": 0.96,
                        "max": 1.05,
                    },
                    "n_cells": {"min": 1, "max": 7, "step": 2},
                    "z0": {"min": 50.0, "max": 600.0, "step": 5.0},
                    "z0_cm": {"min": 30.0, "max": 600.0, "step": 5.0},
                    "tl_length_factor": {
                        "min": 0.8,
                        "max": 1.2,
                    },
                }
            ),
        }
    )

    def _geom(self):
        wavelength = 299.792458 / self.design_freq
        h = 0.5 * wavelength * self.length_factor
        q = 0.5 * h
        n = int(self.n_cells)
        assert n >= 1 and n % 2 == 1, "n_cells must be a positive odd integer"
        yb = [0.0, q] + [q + k * h for k in range(1, n + 1)] + [2 * q + n * h]
        return h, q, n, yb

    def _all_wires_tups(self):
        """The validated all-wires `sterba` geometry at our params."""
        sp = dict(sterba.Builder.default_params)
        for k in ("design_freq", "freq", "base", "length_factor", "n_cells", "spacing"):
            sp[k] = getattr(self, k)
        b = sterba.Builder(sp)
        b.nominal_nsegs = self.nominal_nsegs
        return b.build_wires()

    def _junction_names(self):
        """Map each junction-terminal point -> port name, and list junction
        y positions. Terminals: A-top, B-top, A-bot, B-bot at each interior
        junction (the half-wave-section boundaries)."""
        h, q, n, yb = self._geom()
        top = round(self.base + h, 4)
        bot = round(self.base, 4)
        dx = round(self.spacing, 4)
        jys = [round(yb[k], 4) for k in range(1, n + 2)]  # interior junctions
        jname = {}
        for j, yj in enumerate(jys, 1):
            jname[(0.0, yj, top)] = f"j{j}_At"
            jname[(dx, yj, top)] = f"j{j}_Bt"
            jname[(0.0, yj, bot)] = f"j{j}_Ab"
            jname[(dx, yj, bot)] = f"j{j}_Bb"
        return jys, jname

    def build_wires(self):
        h, q, n, yb = self._geom()
        Ltot = yb[-1]
        _, jname = self._junction_names()
        # Short port-edge: one nominal segment. The DiffTL attaches here, so
        # it must be small relative to the half-wave section.
        pe = h / self.nominal_nsegs

        def key(p):
            return (round(p[0], 4), round(p[1], 4), round(p[2], 4))

        def is_riser(p0, p1):
            ddx = abs(p1[0] - p0[0])
            dy = abs(p1[1] - p0[1])
            dz = abs(p1[2] - p0[2])
            y = 0.5 * (p0[1] + p1[1])
            return dz > 0.5 * h and ddx < 0.01 and dy < 0.01 and 0.01 < y < Ltot - 0.01

        out = []
        for p0, p1, ns, ev in self._all_wires_tups():
            if is_riser(p0, p1):
                continue  # replaced by a DiffTL in build_network()
            name = "feed" if ev is not None else None
            ln = jname.get(key(p0))
            rn = jname.get(key(p1))
            dy = p1[1] - p0[1]
            if abs(dy) > 0.01 and (ln or rn):
                # split this horizontal so its junction end(s) are short
                # named port edges the DiffTL attaches to.
                x, z = p0[0], p0[2]
                s = 1.0 if dy > 0 else -1.0
                cur = p0[1]
                segs = []
                if ln:
                    segs.append((p0[1], p0[1] + s * pe, ln))
                    cur = p0[1] + s * pe
                end = (p1[1] - s * pe) if rn else p1[1]
                segs.append((cur, end, name))
                cur = end
                if rn:
                    segs.append((cur, p1[1], rn))
                for ya, yb_, nm in segs:
                    seglen = abs(yb_ - ya)
                    nseg = (
                        3
                        if (nm and nm != "feed")
                        else max(3, round(self.nominal_nsegs * seglen / h))
                    )
                    out.append(((x, ya, z), (x, yb_, z), nseg, None, nm))
            else:
                out.append((p0, p1, ns, None, name))
        return out

    def build_network(self):
        h, _, n, _ = self._geom()
        jys, _ = self._junction_names()
        ports = {"feed": PortAtEdge("feed")}
        branches = []
        for j in range(1, len(jys) + 1):
            for t in ("At", "Bt", "Ab", "Bb"):
                ports[f"j{j}_{t}"] = PortAtEdge(f"j{j}_{t}")
            # Differential port A = (A-top, B-top), port B = (A-bot, B-bot);
            # the half-wave line phases the sections, the common mode carries
            # the through-current. (Uniform, untwisted polarity is best — a
            # transposed line collapses the array into a loop.)
            branches.append(
                DiffTL(
                    f"j{j}_At",
                    f"j{j}_Bt",
                    f"j{j}_Ab",
                    f"j{j}_Bb",
                    z0=self.z0,
                    length=h * self.tl_length_factor,
                    transposed=False,
                    z0_cm=self.z0_cm,
                )
            )
        return Network(
            ports=ports,
            branches=branches,
            sources=[Driven(port="feed", voltage=1 + 0j)],
        )
