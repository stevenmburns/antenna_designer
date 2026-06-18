"""Physics regression tests for the Cebik (W4RNL) design family.

Each test pins the published Cebik behaviour (resonant impedance, gain,
polarisation/pattern shape) via the PyNEC engine so a geometry regression
in build_wires() is caught. Free-space (ground=None) is used for the
impedance/gain numbers because it removes soil-model dependence; pattern
shape (broadside vs end-on) is checked there too.
"""

from __future__ import annotations

import numpy as np

from antenna_designer.engines import PyNECEngine


def _z(builder, ground=None):
    return PyNECEngine(builder, ground=ground).impedance()[0]


def _far_field(builder, ground=None):
    return PyNECEngine(builder, ground=ground).far_field(
        n_theta=90, n_phi=360, del_theta=1, del_phi=1
    )


# ---------------------------------------------------------------------------
# Half-square
# ---------------------------------------------------------------------------


def test_half_square_resonant_and_low_z():
    """Corner-fed half-square: ~65 ohm, near-resonant at length_factor=1
    (Cebik's max-gain proportions)."""
    from antenna_designer.designs.cebik.half_square import Builder

    z = _z(Builder())
    assert 50.0 < z.real < 80.0
    assert abs(z.imag) < 20.0  # near resonance at the default scale


def test_half_square_gain_matches_cebik():
    """~4.6-4.7 dBi free-space per Cebik's published models."""
    from antenna_designer.designs.cebik.half_square import Builder

    ff = _far_field(Builder())
    assert 4.0 < ff.max_gain < 5.5


def test_half_square_is_broadside_with_end_nulls():
    """Vertically-polarised, bidirectional broadside off +/-x with deep
    nulls off the ends (Cebik: >10 dB side rejection)."""
    from antenna_designer.designs.cebik.half_square import Builder

    ff = _far_field(Builder())
    rings = np.array(ff.rings)  # [theta][phi], dBi
    row = rings[60]  # ~30 deg elevation
    broadside = max(row[0], row[180])
    end_on = max(row[90], row[270])
    assert broadside - end_on > 8.0


def test_half_square_length_factor_tunes_reactance():
    """Reactance climbs monotonically with length_factor through resonance."""
    from antenna_designer.designs.cebik.half_square import Builder

    x_lo = _z(Builder(dict(Builder.default_params, length_factor=0.96))).imag
    x_mid = _z(Builder(dict(Builder.default_params, length_factor=1.00))).imag
    x_hi = _z(Builder(dict(Builder.default_params, length_factor=1.04))).imag
    assert x_lo < x_mid < x_hi


# ---------------------------------------------------------------------------
# Bobtail curtain
# ---------------------------------------------------------------------------


def test_bobtail_gain_exceeds_half_square():
    """Three-element curtain: ~5+ dBi broadside, more than the half-square's
    ~4.7 (Cebik: ~5.1-5.2 dBi)."""
    from antenna_designer.designs.cebik.bobtail import Builder

    ff = _far_field(Builder())
    assert ff.max_gain > 5.0


def test_bobtail_broadside_directivity():
    """Vertically-polarised, sharply bidirectional broadside off +/-x with
    very deep end nulls (3 in-phase verticals)."""
    from antenna_designer.designs.cebik.bobtail import Builder

    ff = _far_field(Builder())
    rings = np.array(ff.rings)
    row = rings[60]
    broadside = max(row[0], row[180])
    end_on = max(row[90], row[270])
    assert broadside - end_on > 20.0


def test_bobtail_feed_is_high_impedance():
    """Base feed of the centre element is a high, reactive point (Cebik:
    hundreds to thousands of ohms, tank-matched)."""
    from antenna_designer.designs.cebik.bobtail import Builder

    z = _z(Builder())
    assert z.real > 500.0
    assert abs(z.imag) > 1000.0


def test_bobtail_only_centre_element_is_fed():
    """Exactly one driven gap; the outer verticals are passive."""
    from antenna_designer.designs.cebik.bobtail import Builder

    feeds = [t for t in Builder().build_wires() if t[3] is not None]
    assert len(feeds) == 1
    # The fed gap sits on the centre vertical (y = 0).
    (x0, y0, _), (x1, y1, _), _, _ = feeds[0]
    assert y0 == 0.0 and y1 == 0.0


# ---------------------------------------------------------------------------
# Cubical quad beam
# ---------------------------------------------------------------------------


def test_quad_forward_gain():
    """~7 dBi forward (Cebik: 6.6-7.5 dBi for the wideband 2-el quad)."""
    from antenna_designer.designs.cebik.quad import Builder

    ff = _far_field(Builder())
    assert ff.max_gain > 6.5


def test_quad_driver_near_resonant():
    """Driver loop ~1.01 wl is near resonance at the default scale."""
    from antenna_designer.designs.cebik.quad import Builder

    z = _z(Builder())
    assert abs(z.imag) < 35.0


def test_quad_fires_toward_driver_with_front_to_back():
    """Beam fires +x (toward the driver, away from the reflector at -x)."""
    from antenna_designer.designs.cebik.quad import Builder

    ff = _far_field(Builder())
    rings = np.array(ff.rings)
    front = rings[:, 0].max()  # +x
    back = rings[:, 180].max()  # -x
    assert front - back > 6.0


def test_quad_has_two_loops_one_fed():
    """Reflector (passive) + driver (one fed gap) = 2 four-sided loops."""
    from antenna_designer.designs.cebik.quad import Builder

    tups = Builder().build_wires()
    feeds = [t for t in tups if t[3] is not None]
    assert len(feeds) == 1
    # Reflector sits behind the driver (more negative x).
    xs = sorted({round(t[0][0], 6) for t in tups})
    assert len(xs) == 2 and xs[0] < xs[1]


# ---------------------------------------------------------------------------
# Lazy-H
# ---------------------------------------------------------------------------


def test_lazy_h_stacking_gain():
    """Two stacked in-phase 1 wl elements give ~8 dBi free-space -- well
    above a single ~1 wl element's ~4 dBi (the vertical-stacking gain)."""
    from antenna_designer.designs.cebik.lazy_h import Builder

    ff = _far_field(Builder())
    assert ff.max_gain > 7.0


def test_lazy_h_broadside_horizontal():
    """Bidirectional broadside off +/-x with deep end nulls."""
    from antenna_designer.designs.cebik.lazy_h import Builder

    ff = _far_field(Builder())
    rings = np.array(ff.rings)
    row = rings[60]
    broadside = max(row[0], row[180])
    end_on = max(row[90], row[270])
    assert broadside - end_on > 15.0


def test_lazy_h_two_in_phase_feeds():
    """Two centre feeds, both at y=0, both driven in phase (1+0j); by
    symmetry they present equal feed impedance."""
    from antenna_designer.designs.cebik.lazy_h import Builder

    feeds = [t for t in Builder().build_wires() if t[3] is not None]
    assert len(feeds) == 2
    assert all(f[3] == 1 + 0j for f in feeds)
    assert all(f[0][1] == -0.05 and f[1][1] == 0.05 for f in feeds)
    zs = PyNECEngine(Builder(), ground=None).impedance()
    assert abs(zs[0] - zs[1]) < 1.0  # symmetric -> equal


def test_lazy_h_wider_spacing_adds_gain():
    """Expanding the stack toward ~5/8 wl raises gain (W2EEY expansion)."""
    from antenna_designer.designs.cebik.lazy_h import Builder

    g_half = _far_field(
        Builder(dict(Builder.default_params, spacing_frac=0.5))
    ).max_gain
    g_wide = _far_field(
        Builder(dict(Builder.default_params, spacing_frac=0.625))
    ).max_gain
    assert g_wide > g_half
