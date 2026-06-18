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


# ---------------------------------------------------------------------------
# LPDA (log-periodic dipole array)
# ---------------------------------------------------------------------------


def test_lpda_broadband_forward_gain():
    """The defining LPDA behaviour: ~6-9 dBi forward gain held across a wide
    band, firing toward the apex (+x). (Feedpoint impedance is not asserted
    -- the ideal lossless crossed feeder makes it unreliable; see module
    docstring.)"""
    from antenna_designer.designs.cebik.lpda import Builder

    for fr in (24.0, 26.0, 28.57, 30.0):
        b = Builder(dict(Builder.default_params, freq=fr))
        ff = _far_field(b)
        rings = np.array(ff.rings)
        front = rings[:, 0].max()  # +x, toward the apex
        back = rings[:, 180].max()
        assert ff.max_gain > 5.5, (fr, ff.max_gain)
        assert front > back, (fr, front, back)


def test_lpda_elements_scale_by_tau():
    """Element half-lengths form a geometric sequence with ratio tau."""
    from antenna_designer.designs.cebik.lpda import Builder

    b = Builder()
    half, x = b._layout()
    ratios = [half[k + 1] / half[k] for k in range(len(half) - 1)]
    assert all(abs(r - b.tau) < 1e-9 for r in ratios)
    # boom positions strictly increase toward the front
    assert all(x[k + 1] > x[k] for k in range(len(x) - 1))


def test_lpda_feeder_is_crossed_and_front_driven():
    """Every feeder section is crossed (negative z0) and the source sits on
    the front (shortest) element."""
    from antenna_designer.designs.cebik.lpda import Builder
    from antenna_designer.network import TL, Driven

    b = Builder()
    net = b.build_network()
    tls = [br for br in net.branches if isinstance(br, TL)]
    assert len(tls) == b.n_elements - 1
    assert all(tl.transposed and tl.z0 > 0 for tl in tls)  # all crossed
    (src,) = net.sources
    assert isinstance(src, Driven)
    assert src.port == f"d{b.n_elements - 1}"  # frontmost / shortest


# ---------------------------------------------------------------------------
# HB9CV / ZL-Special (2-element all-driven phased beam)
# ---------------------------------------------------------------------------


def test_hb9cv_forward_gain_and_endfire():
    """~6-7 dBi (like a 2-el Yagi) firing toward the front (+x). F/B is real
    but shallow in this ideal-crossed-TL model -- see module docstring."""
    from antenna_designer.designs.cebik.hb9cv import Builder

    ff = _far_field(Builder())
    rings = np.array(ff.rings)
    front = rings[:, 0].max()
    back = rings[:, 180].max()
    assert ff.max_gain > 6.0
    assert front - back > 5.0


def test_hb9cv_feed_resistive_inductive():
    """Cebik: feed ~40-55 ohm resistive with inductive reactance. Both
    elements are driven through a single crossed phasing line."""
    from antenna_designer.designs.cebik.hb9cv import Builder

    z = _z(Builder())
    assert z.real > 15.0  # positive, real driving-point R
    assert z.imag > 0.0  # inductive (needs series-cap cancellation)


def test_hb9cv_both_driven_via_one_crossed_line():
    """No parasite: a single crossed (transposed) phasing line couples the
    two driven element centres; the source sits on the front element."""
    from antenna_designer.designs.cebik.hb9cv import Builder
    from antenna_designer.network import TL, Driven

    net = Builder().build_network()
    tls = [br for br in net.branches if isinstance(br, TL)]
    assert len(tls) == 1 and tls[0].transposed and tls[0].z0 > 0
    assert {tls[0].a, tls[0].b} == {"rear", "front"}
    (src,) = net.sources
    assert isinstance(src, Driven) and src.port == "front"


# ---------------------------------------------------------------------------
# Terminated rhombic
# ---------------------------------------------------------------------------


def test_rhombic_unidirectional_when_terminated():
    """The terminating resistor makes the traveling-wave pattern
    unidirectional toward the terminated apex (+x)."""
    from antenna_designer.designs.cebik.rhombic import Builder

    ff = _far_field(Builder())
    rings = np.array(ff.rings)
    front = rings[:, 0].max()  # +x toward termination
    back = rings[:, 180].max()
    assert ff.max_gain > 6.0
    assert front - back > 12.0


def test_rhombic_termination_creates_the_directivity():
    """Remove the termination (R -> huge) and the F/B collapses: the
    progressive wave is gone and the pattern goes ~bidirectional."""
    from antenna_designer.designs.cebik.rhombic import Builder

    def fb(r):
        b = Builder(dict(Builder.default_params, term_r=r))
        rings = np.array(_far_field(b).rings)
        return rings[:, 0].max() - rings[:, 180].max()

    assert fb(700.0) > 12.0
    assert fb(1e9) < 5.0


def test_rhombic_impedance_tracks_termination():
    """Traveling-wave antenna: the driving-point R sits near the
    termination value, and it scales with it (broadband behaviour)."""
    from antenna_designer.designs.cebik.rhombic import Builder

    z600 = _z(Builder(dict(Builder.default_params, term_r=600.0)))
    z800 = _z(Builder(dict(Builder.default_params, term_r=800.0)))
    assert 450.0 < z600.real < 750.0
    assert z800.real > z600.real  # tracks the termination upward


def test_rhombic_has_terminating_load_and_feed():
    """One driven feed apex and one resistive Load at the far apex."""
    from antenna_designer.designs.cebik.rhombic import Builder
    from antenna_designer.network import Driven, Load

    net = Builder().build_network()
    loads = [br for br in net.branches if isinstance(br, Load)]
    assert len(loads) == 1
    assert loads[0].port == "term" and loads[0].r == 700.0
    (src,) = net.sources
    assert isinstance(src, Driven) and src.port == "feed"


# ---------------------------------------------------------------------------
# T2FD (terminated tilted folded dipole)
# ---------------------------------------------------------------------------


def _swr(z, z0):
    g = abs((z - z0) / (z + z0))
    return (1 + g) / (1 - g)


_T2FD_BAND = (14.0, 18.0, 22.0, 28.57, 36.0, 45.0, 56.0)


def test_t2fd_broadband_low_swr():
    """The defining T2FD behaviour: a flat SWR curve over a 4:1 frequency
    range (here referenced to the ~850 ohm the terminated geometry settles
    to), unlike a resonant antenna."""
    from antenna_designer.designs.cebik.t2fd import Builder

    z0 = 850.0
    swrs = [
        _swr(_z(Builder(dict(Builder.default_params, freq=f))), z0) for f in _T2FD_BAND
    ]
    assert max(swrs) < 2.5, dict(zip(_T2FD_BAND, swrs))


def test_t2fd_termination_flattens_the_response():
    """Removing the resistor (R -> huge) restores sharp resonances: the
    unterminated max-SWR over the band is far worse than terminated."""
    from antenna_designer.designs.cebik.t2fd import Builder

    z0 = 850.0

    def band_max(r):
        return max(
            _swr(_z(Builder(dict(Builder.default_params, freq=f, term_r=r))), z0)
            for f in _T2FD_BAND
        )

    assert band_max(820.0) < 2.5
    assert band_max(1e9) > 10.0  # huge anti-resonant spike without the load


def test_t2fd_gain_is_reduced_by_loss():
    """Power burned in the terminating resistor drops gain below a resonant
    dipole's ~2.1 dBi -- the bandwidth/efficiency trade."""
    from antenna_designer.designs.cebik.t2fd import Builder

    ff = _far_field(Builder())
    assert ff.max_gain < 2.0


def test_t2fd_folded_with_termination():
    """Folded pair (two end shorts), one driven feed, one resistive Load."""
    from antenna_designer.designs.cebik.t2fd import Builder
    from antenna_designer.network import Driven, Load

    tups = Builder().build_wires()
    feeds = [t for t in tups if len(t) == 5 and t[4] == "feed"]
    terms = [t for t in tups if len(t) == 5 and t[4] == "term"]
    assert len(feeds) == 1 and len(terms) == 1
    net = Builder().build_network()
    (load,) = [br for br in net.branches if isinstance(br, Load)]
    assert load.port == "term"
    (src,) = net.sources
    assert isinstance(src, Driven) and src.port == "feed"
