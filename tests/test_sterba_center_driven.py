"""Tests for the center-fed-section Sterba variant (wire.sterba_center_driven).

The design deletes the 8 interior risers and exposes the center of each of the
10 horizontal sections as a feedpoint. These check the build/solve mechanism;
the gain result (≈ the all-wires reference, once each section's net moment is
matched) is measured by scripts/sterba_center_driven_experiment.py.
"""

import numpy as np

from antennaknobs.designs.wire import sterba_center_driven
from antennaknobs.engines import MomwireEngine

FF_KW = dict(n_theta=90, n_phi=360, del_theta=1, del_phi=1)


def test_ten_sections_named_edges_match_ports():
    b = sterba_center_driven.Builder()
    specs = b.section_specs()
    assert len(specs) == 10  # n_cells=3: 5 horizontal sections per conductor × 2
    named = {t[4] for t in b.build_wires() if len(t) >= 5 and t[4]}
    assert named == set(b.build_network().ports) == {nm for nm, _c in specs}


def test_section_feeds_sit_at_section_centers():
    """Each named edge is short and centered on its section midpoint (so the
    delta-gap feed drives the section at its current antinode for half-waves)."""
    b = sterba_center_driven.Builder()
    centers = dict(b.section_specs())
    for t in b.build_wires():
        if len(t) >= 5 and t[4]:
            mid_y = 0.5 * (t[0][1] + t[1][1])
            assert abs(mid_y - centers[t[4]][1]) < 1e-6


def test_builds_and_solves_to_finite_z_and_gain():
    eng = MomwireEngine(sterba_center_driven.Builder(), ground=None)
    zs = eng.impedance()
    assert len(zs) == 10
    assert all(np.isfinite(z.real) and np.isfinite(z.imag) for z in zs)
    assert np.isfinite(eng.far_field(**FF_KW).max_gain)
