"""Sterba curtain, center-fed-section variant.

Like `sterba_driven`, this deletes the 8 interior risers — but instead of
feeding the section *ends* (the junction ports, which become free wire ends
where current must vanish), it puts one feedpoint at the **center** of each
horizontal section. The riser-mask experiment proved the far field is set
entirely by the horizontal current distribution; an isolated half-wave
section's natural center-fed mode is a co-phased half-sine, which is exactly
what the reference sections look like. So driving each section's center with
the reference's center current (amplitude + phase) should reproduce the
reference distribution — and thus the ~10.5 dBi gain — if the verticals are
purely a phasing mechanism.

As with `sterba_driven`, voltages are supplied via `feed_voltages`; the driver
script solves V = Y⁻¹·I_target so the section-center currents match the
reference. End risers are kept (they are not among the 8 inner verticals).
PysimEngine-only.
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
            "length_factor": 1.0,  # match the all-wires reference geometry
            "n_cells": 3,
            "spacing": 0.04,
            # Per-section drive voltages, {section_name -> complex}. None ->
            # placeholder 1+0j (used to extract Y; the script solves for the
            # real voltages).
            "feed_voltages": None,
            "ui_params": MappingProxyType(
                {
                    "target_z0": 75.0,
                    "sweep_policy": {"band_locked": True},
                    "length_factor": {
                        "min": 0.96,
                        "max": 1.05,
                        "step": 0.001,
                        "precision": 4,
                    },
                    "n_cells": {"min": 1, "max": 7, "step": 2},
                }
            ),
        }
    )

    def _is_interior_riser(self, p0, p1):
        h, _, _, yb = self._geom()
        Ltot = yb[-1]
        ddx = abs(p1[0] - p0[0])
        dy = abs(p1[1] - p0[1])
        dz = abs(p1[2] - p0[2])
        y = 0.5 * (p0[1] + p1[1])
        return dz > 0.5 * h and ddx < 0.01 and dy < 0.01 and 0.01 < y < Ltot - 0.01

    def section_specs(self):
        """Ordered [(name, center_xyz)] for each horizontal section of the
        riser-less curtain — the same iteration order build_wires uses, so the
        names line up. Center is the section midpoint (the current antinode)."""
        specs = []
        i = 0
        for p0, p1, _ns, _ev in self._all_wires_tups():
            if self._is_interior_riser(p0, p1):
                continue
            is_horiz = abs(p0[2] - p1[2]) < 1e-9 and abs(p1[1] - p0[1]) > 0.01
            if is_horiz:
                c = (
                    0.5 * (p0[0] + p1[0]),
                    0.5 * (p0[1] + p1[1]),
                    0.5 * (p0[2] + p1[2]),
                )
                specs.append((f"s{i}", c))
                i += 1
        return specs

    def build_wires(self):
        h, _, _, _ = self._geom()
        pe = h / self.nominal_nsegs
        out = []
        i = 0
        for p0, p1, ns, _ev in self._all_wires_tups():
            if self._is_interior_riser(p0, p1):
                continue  # one of the 8 inner risers — deleted
            is_horiz = abs(p0[2] - p1[2]) < 1e-9 and abs(p1[1] - p0[1]) > 0.01
            if is_horiz:
                name = f"s{i}"
                i += 1
                x, z = p0[0], p0[2]
                cy = 0.5 * (p0[1] + p1[1])
                s = 1.0 if p1[1] > p0[1] else -1.0
                a, b = cy - s * pe / 2, cy + s * pe / 2
                # left | short named center edge | right
                for ya, yb_, nm in ((p0[1], a, None), (a, b, name), (b, p1[1], None)):
                    seglen = abs(yb_ - ya)
                    nseg = 3 if nm else max(3, round(self.nominal_nsegs * seglen / h))
                    out.append(((x, ya, z), (x, yb_, z), nseg, None, nm))
            else:
                out.append((p0, p1, ns, None, None))
        return out

    def build_network(self):
        fv = dict(self.feed_voltages) if self.feed_voltages else {}
        ports = {}
        sources = []
        for name, _c in self.section_specs():
            ports[name] = PortAtEdge(name)
            sources.append(Driven(port=name, voltage=fv.get(name, 1 + 0j)))
        return Network(ports=ports, branches=[], sources=sources)
