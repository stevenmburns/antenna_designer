"""Experiment: center-feed each riser-less Sterba section so its net current
moment matches the all-wires reference — and recover the gain.

Counterpart to sterba_driven_experiment.py (which fed the section *ends* — free
wire-ends where current must vanish — and reached only ~3 dBi). Here each of the
10 horizontal sections is fed at its center.

The quantity that sets the broadside field is each section's net current moment
M_y = ∫ I·dy (at broadside the curtain's wires are all at x≈0, so the field is
∝ |Σ M_y|). So we match *that*, not the midpoint current: matching the midpoint
current gets the half-wave sections right but flips the quarter-wave end
sections 180° (their midpoint isn't their antinode), costing ~3 dB. Because the
far field is linear in the feed voltages, M_y = G·V for a section-moment
response matrix G, so we solve V = G⁻¹·M_y,ref.

    .venv/bin/python scripts/sterba_center_driven_experiment.py [out.png]
"""

import sys

import numpy as np

from antenna_designer.designs.freq_based import sterba, sterba_center_driven
from antenna_designer.engines import PysimEngine
from antenna_designer.far_field import plot_patterns

FF_KW = dict(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
GROUND = None  # free space, vs the reference's 10.501 dBi


def _broadside_coherence(cd):
    """|Σ I·dy| / Σ|I·dy| over horizontal segments: ~1 co-phased, ~0 scattered."""
    num, den = 0 + 0j, 0.0
    for wc in cd:
        P, I = wc.knot_positions, wc.knot_currents
        for k in range(len(P) - 1):
            d = P[k + 1] - P[k]
            if abs(d[1]) > max(abs(d[0]), abs(d[2])):
                m = 0.5 * (I[k] + I[k + 1]) * d[1]
                num += m
                den += abs(m)
    return abs(num) / den if den else float("nan")


def _section_extents(b):
    """[(name, x, z, y0, y1)] for each horizontal section of the riser-less
    curtain (same order as section_specs)."""
    out = []
    i = 0
    for p0, p1, _ns, _ev in b._all_wires_tups():
        if b._is_interior_riser(p0, p1):
            continue
        if abs(p0[2] - p1[2]) < 1e-9 and abs(p1[1] - p0[1]) > 0.01:
            out.append(
                (f"s{i}", 0.5 * (p0[0] + p1[0]), 0.5 * (p0[2] + p1[2]), p0[1], p1[1])
            )
            i += 1
    return out


def _section_moment(cd, x, z, y0, y1):
    """Net y-current moment M_y = Σ I·dy of the horizontal segments belonging
    to one section (binned by x, z and y-range)."""
    my = 0 + 0j
    lo, hi = min(y0, y1) - 0.3, max(y0, y1) + 0.3
    for wc in cd:
        P, I = wc.knot_positions, wc.knot_currents
        for k in range(len(P) - 1):
            d = P[k + 1] - P[k]
            m = 0.5 * (P[k] + P[k + 1])
            if (
                abs(d[1]) > max(abs(d[0]), abs(d[2]))
                and abs(m[2] - z) < 0.5
                and abs(m[0] - x) < 0.03
                and lo < m[1] < hi
            ):
                my += 0.5 * (I[k] + I[k + 1]) * d[1]
    return my


def _moments(cd, exts):
    return np.array([_section_moment(cd, x, z, y0, y1) for _n, x, z, y0, y1 in exts])


def _drive(builder_cls, ports, V):
    p = dict(builder_cls.default_params)
    p["feed_voltages"] = {nm: complex(V[i]) for i, nm in enumerate(ports)}
    return PysimEngine(builder_cls(p), ground=GROUND)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "sterba_center_driven_pattern.png"

    eng_ref = PysimEngine(sterba.Builder(), ground=GROUND)
    ff_ref = eng_ref.far_field(**FF_KW)
    cd_ref = eng_ref.current_distribution()

    b0 = sterba_center_driven.Builder()
    exts = _section_extents(b0)
    ports = [nm for nm, _c in b0.section_specs()]
    n = len(ports)

    # Reference per-section net moments (the match target).
    My_ref = _moments(cd_ref, exts)

    # Section-moment response matrix: G[:, j] = moments when only port j is at 1 V.
    G = np.empty((n, n), dtype=np.complex128)
    for j in range(n):
        e_j = np.zeros(n, dtype=np.complex128)
        e_j[j] = 1.0
        G[:, j] = _moments(
            _drive(sterba_center_driven.Builder, ports, e_j).current_distribution(),
            exts,
        )

    V = np.linalg.solve(G, My_ref)
    eng = _drive(sterba_center_driven.Builder, ports, V)
    cd = eng.current_distribution()
    ff = eng.far_field(**FF_KW)

    moment_err = np.max(np.abs(_moments(cd, exts) - My_ref)) / np.max(np.abs(My_ref))
    print(f"sections fed (at center)     : {n}")
    print(f"cond(G)                      : {np.linalg.cond(G):.2e}")
    print(f"section net-moment match err : {moment_err:.2e} (M_y = G·V round-trip)")
    print(
        f"broadside coherence (driven) : {_broadside_coherence(cd):.3f} "
        f"(reference {_broadside_coherence(cd_ref):.3f}; 1=co-phased)"
    )
    print(f"max_gain (center-driven)     : {ff.max_gain:.3f} dBi")
    print(f"max_gain (all-wires ref)     : {ff_ref.max_gain:.3f} dBi")
    print(f"gap                          : {ff.max_gain - ff_ref.max_gain:+.3f} dB")

    plot_patterns(
        [ff_ref.rings, ff.rings],
        ["all wires", "center-driven"],
        ff_ref.thetas,
        ff_ref.phis,
        fn=out,
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
