"""Sterba curtain, explicit-feedpoint variant.

A research variant of `freq_based.sterba` that, like `sterba_difftl`, removes
the 8 interior vertical riser wires — but instead of stitching each vertical
*pair* back together with a `DiffTL`, it leaves the horizontal sections
electrically disconnected and drives them **directly**: an explicit feedpoint
is placed at each of the 4 ports a vertical pair exposes at a junction —
two top (`j_At`, `j_Bt`) and two bottom (`j_Ab`, `j_Bb`).

The experiment: are the inner verticals purely a phasing mechanism? If so,
deleting them and driving each section's junction-ends with the *current*
(amplitude + phase) the all-wires reference carried there should reproduce the
reference's far field. The reference is a continuous wire with no source at
these points, so "matching the reference" means matching its **port currents**,
not applying its numbers as voltages. Since the system is linear (I = Y·V), the
driver script solves V = Y⁻¹·I_target and feeds those voltages back in via
`feed_voltages` — this design just exposes the ports and applies whatever
voltages it is handed.

Build it up one junction at a time with `active_junctions`: a single central
pair first (to confirm the feedpoint mechanism solves at all), then more, up to
all n_cells+1. Sections at inactive junctions stay as plain floating (parasitic)
wires. PysimEngine-only (multi-feed network, like `sterba_difftl`).
"""

from ...network import Driven, Network, PortAtEdge
from . import sterba_difftl

from types import MappingProxyType


class Builder(sterba_difftl.Builder):
    default_params = MappingProxyType(
        {
            "design_freq": 28.47,
            "freq": 28.47,
            "base": 7.0,
            # Match the all-wires reference geometry exactly (it runs at 1.0),
            # so the reference currents we inject correspond to identical
            # horizontal sections. No DiffTL here, so the ideal-line
            # half-wave singularity that forced difftl to 0.99 doesn't apply.
            "length_factor": 1.0,
            "n_cells": 3,  # odd; interior junctions (= riser pairs) = n_cells + 1
            "spacing": 0.04,
            # Which junctions (1..n_cells+1) get explicit feedpoints. None = all.
            # The incremental knob: start [2] (one central pair), then grow.
            "active_junctions": None,
            # Per-port drive voltages, {port_name -> complex}. Empty/None ->
            # placeholder 1+0j on every active port (used only to extract the
            # multi-port Y; the script then solves for the real voltages).
            "feed_voltages": None,
            "ui_params": MappingProxyType(
                {
                    "target_z0": 75.0,
                    "sweep_policy": {"band_locked": True},
                    "length_factor": {
                        "min": 0.96,
                        "max": 1.05,
                    },
                    "n_cells": {"min": 1, "max": 7, "step": 2},
                }
            ),
        }
    )

    def _active_set(self):
        """Set of active junction indices (1..n_cells+1)."""
        _, _, n, _ = self._geom()
        if self.active_junctions is None:
            return set(range(1, n + 2))
        active = {int(j) for j in self.active_junctions}
        assert active <= set(range(1, n + 2)), (
            f"active_junctions {sorted(active)} out of range 1..{n + 1}"
        )
        return active

    def _active_jname(self):
        """Junction-terminal point -> port name, restricted to active junctions."""
        _, jname = self._junction_names()
        active = self._active_set()
        return {
            pt: nm for pt, nm in jname.items() if int(nm[1:].split("_")[0]) in active
        }

    def build_wires(self):
        """Riser-less geometry. All 8 interior risers are removed; only the
        junction terminals of *active* junctions are split into short named
        port edges. No central feed — every section is driven solely through
        its junction-end ports."""
        h, q, n, yb = self._geom()
        Ltot = yb[-1]
        jname = self._active_jname()
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
        for p0, p1, ns, _ev in self._all_wires_tups():
            if is_riser(p0, p1):
                continue  # one of the 8 inner risers — deleted
            ln = jname.get(key(p0))
            rn = jname.get(key(p1))
            dy = p1[1] - p0[1]
            if abs(dy) > 0.01 and (ln or rn):
                # Split this horizontal so its active-junction end(s) become
                # short named port edges the feedpoint attaches to.
                x, z = p0[0], p0[2]
                s = 1.0 if dy > 0 else -1.0
                cur = p0[1]
                segs = []
                if ln:
                    segs.append((p0[1], p0[1] + s * pe, ln))
                    cur = p0[1] + s * pe
                end = (p1[1] - s * pe) if rn else p1[1]
                segs.append((cur, end, None))
                cur = end
                if rn:
                    segs.append((cur, p1[1], rn))
                for ya, yb_, nm in segs:
                    seglen = abs(yb_ - ya)
                    nseg = 3 if nm else max(3, round(self.nominal_nsegs * seglen / h))
                    out.append(((x, ya, z), (x, yb_, z), nseg, None, nm))
            else:
                out.append((p0, p1, ns, None, None))
        return out

    def build_network(self):
        """One driven feedpoint per port of each active junction. No branches:
        a pure multi-feed network (NetworkReducer pins driven ports at their
        applied voltage and floats the rest)."""
        active = sorted(self._active_set())
        fv = dict(self.feed_voltages) if self.feed_voltages else {}
        ports = {}
        sources = []
        for j in active:
            for t in ("At", "Bt", "Ab", "Bb"):
                name = f"j{j}_{t}"
                ports[name] = PortAtEdge(name)
                sources.append(Driven(port=name, voltage=fv.get(name, 1 + 0j)))
        return Network(ports=ports, branches=[], sources=sources)
