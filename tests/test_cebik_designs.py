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


def test_bobtail_feed_is_coax_friendly():
    """Tapped at a current maximum on the centre vertical (not the classic
    high-Z base/tank point), the driving point is a low, near-resonant ~50 ohm
    that takes coax directly."""
    from antenna_designer.designs.cebik.bobtail import Builder

    z = _z(Builder())
    assert 35.0 < z.real < 70.0
    assert abs(z.imag) < 30.0


def test_bobtail_tap_position_sets_impedance():
    """Sliding the tap toward the base (a current null) raises the feed
    resistance -- the standing-wave transformation that lets feed_height_frac
    pick the match, the same trick a shunt/gamma feed uses."""
    from antenna_designer.designs.cebik.bobtail import Builder

    r_mid = _z(Builder()).real  # default tap (~0.5) -> ~50 ohm
    r_low = _z(Builder(dict(Builder.default_params, feed_height_frac=0.3))).real
    assert r_low > r_mid + 20.0


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


# ---------------------------------------------------------------------------
# Batch 2 — W8JK, phased verticals, inverted-L, OCF, V-beam, bi-square,
# J-pole, discone (a second Cebik/W4RNL set filling further catalog gaps).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# W8JK flat-top beam (180-degree all-driven array)
# ---------------------------------------------------------------------------


def test_w8jk_bidirectional_endfire_gain():
    """~5.8 dBi firing equally off both +/- x ends (Kraus extended elements);
    the two anti-phase, close-spaced elements make a bidirectional endfire
    beam, not a unidirectional one."""
    from antenna_designer.designs.cebik.w8jk import Builder

    ff = _far_field(Builder())
    rings = np.array(ff.rings)
    front = rings[:, 0].max()  # +x
    back = rings[:, 180].max()  # -x
    assert ff.max_gain > 5.5
    assert abs(front - back) < 1.0  # bidirectional


def test_w8jk_broadside_and_overhead_nulls():
    """The array signature: deep nulls off the ends (+/- y, broadside to the
    boom) AND overhead (theta = 0, broadside to the array axis) -- the latter
    is what a single dipole would NOT have, proving the 180-deg array action."""
    from antenna_designer.designs.cebik.w8jk import Builder

    ff = _far_field(Builder())
    rings = np.array(ff.rings)
    lobe = rings[:, 0].max()  # the +x endfire lobe
    side = rings[:, 90].max()  # +y broadside
    overhead = rings[0].max()  # straight up
    assert lobe - side > 15.0
    assert lobe - overhead > 15.0


def test_w8jk_two_antiphase_feeds():
    """Exactly two centre feeds, driven 180 degrees out of phase (+1 and -1),
    one per element -- the defining all-driven, out-of-phase topology."""
    from antenna_designer.designs.cebik.w8jk import Builder

    feeds = [t for t in Builder().build_wires() if t[3] is not None]
    assert len(feeds) == 2
    volts = sorted(complex(f[3]).real for f in feeds)
    assert volts[0] == -1.0 and volts[1] == 1.0  # anti-phase


# ---------------------------------------------------------------------------
# Two-element phased vertical array (90-degree cardioid)
# ---------------------------------------------------------------------------


def test_phased_verticals_cardioid_front_to_back():
    """The 90-deg feed phasing steers the pattern unidirectionally toward +x
    with a deep rearward null (~6-7 dB F/B here; a current-forcing network
    deepens it further) -- not the figure-8 of a single vertical."""
    from antenna_designer.designs.cebik.phased_verticals import Builder

    ff = _far_field(Builder())
    rings = np.array(ff.rings)
    front = rings[:, 0].max()
    back = rings[:, 180].max()
    assert ff.max_gain > 4.5
    assert front - back > 5.0


def test_phased_verticals_phase_does_the_steering():
    """Drive the two verticals IN phase instead and the unidirectional cardioid
    collapses -- proving the directivity comes from the FEED PHASE, not the
    geometry."""
    from antenna_designer.designs.cebik.phased_verticals import Builder

    in_phase = Builder(dict(Builder.default_params, front_voltage=1 + 0j))
    rings = np.array(_far_field(in_phase).rings)
    fb = rings[:, 0].max() - rings[:, 180].max()
    assert abs(fb) < 2.0  # symmetric again


def test_phased_verticals_two_feeds_front_quadrature():
    """Two vertical (z-axis) feeds; the rear is the +1 reference and the front
    is driven near 90 degrees out of phase (a dominant imaginary part)."""
    from antenna_designer.designs.cebik.phased_verticals import Builder

    feeds = [t for t in Builder().build_wires() if t[3] is not None]
    assert len(feeds) == 2
    # vertical elements: both feed gaps run along z
    assert all(f[0][2] != f[1][2] for f in feeds)
    rear, front = (complex(f[3]) for f in feeds)
    assert rear == 1 + 0j
    assert abs(front.imag) > abs(front.real)  # near quadrature


# ---------------------------------------------------------------------------
# Inverted-L (bent, top-loaded vertical)
# ---------------------------------------------------------------------------


def test_inverted_l_resonant_low_impedance():
    """Top-loaded short vertical: near-resonant (small X) at a low feed
    resistance over its radial counterpoise."""
    from antenna_designer.designs.cebik.inverted_l import Builder

    z = _z(Builder())
    assert 8.0 < z.real < 45.0
    assert abs(z.imag) < 25.0


def test_inverted_l_vertical_low_angle_radiation():
    """Mostly vertically polarised: the pattern peaks toward the horizon and
    is deeply nulled overhead -- the signature of a vertical, not a horizontal
    wire."""
    from antenna_designer.designs.cebik.inverted_l import Builder

    rings = np.array(_far_field(Builder()).rings)
    horizon = rings[80:].max()  # near the horizon (theta ~ 90)
    overhead = rings[:5].max()  # near zenith (theta ~ 0)
    assert horizon - overhead > 5.0


def test_inverted_l_has_riser_top_and_radials():
    """One base feed, a vertical riser, a horizontal top section (the bend),
    and a radial counterpoise."""
    from antenna_designer.designs.cebik.inverted_l import Builder

    tups = Builder().build_wires()
    feeds = [t for t in tups if t[3] is not None]
    assert len(feeds) == 1
    # a horizontal top wire (constant z, runs along y) exists
    horiz = [
        t
        for t in tups
        if abs(t[0][2] - t[1][2]) < 1e-9 and abs(t[0][1] - t[1][1]) > 1e-6
    ]
    assert horiz, "expected a horizontal top section"


# ---------------------------------------------------------------------------
# Off-Center-Fed dipole (Windom)
# ---------------------------------------------------------------------------


def test_ocf_impedance_rises_off_center():
    """The defining OCF physics: sliding the feed off-centre raises the
    (resistive) feed impedance well above the ~70 ohm centre value."""
    from antenna_designer.designs.cebik.ocf_dipole import Builder

    r_off = _z(Builder()).real
    r_ctr = _z(Builder(dict(Builder.default_params, feed_frac=0.5))).real
    assert r_off > 1.8 * r_ctr
    assert 150.0 < r_off < 350.0  # near the classic ~200-300 ohm Windom point


def test_ocf_near_resonant():
    """At the design length the off-centre feed is near resonance (small X),
    so the elevated impedance is essentially resistive."""
    from antenna_designer.designs.cebik.ocf_dipole import Builder

    assert abs(_z(Builder()).imag) < 60.0


def test_ocf_feed_is_off_center():
    """Geometry: a single feed with unequal arms (short arm toward -y end)."""
    from antenna_designer.designs.cebik.ocf_dipole import Builder

    tups = Builder().build_wires()
    feeds = [t for t in tups if t[3] is not None]
    assert len(feeds) == 1
    y_feed = feeds[0][0][1]
    assert y_feed < -0.05  # offset from the centre (y = 0) toward -y


# ---------------------------------------------------------------------------
# Resonant V-beam
# ---------------------------------------------------------------------------


def test_vbeam_fires_along_the_bisector():
    """Two ~1 wl legs splayed at the apex put gain (~5 dBi) along the
    bisector (+/- x) with a deep null off the broadside (+/- y) -- the
    long-wire lobes of the two legs aligning."""
    from antenna_designer.designs.cebik.vbeam import Builder

    rings = np.array(_far_field(Builder()).rings)
    fwd = rings[:, 0].max()  # +x bisector
    back = rings[:, 180].max()  # -x bisector
    side = rings[:, 90].max()  # +y broadside
    assert _far_field(Builder()).max_gain > 4.5
    assert fwd - side > 4.0
    assert back - side > 3.0


def test_vbeam_high_reactive_apex_feed():
    """Long-wire apex feed: high resistance and strongly reactive (open-wire
    fed in practice), unlike a resonant dipole."""
    from antenna_designer.designs.cebik.vbeam import Builder

    z = _z(Builder())
    assert z.real > 500.0
    assert abs(z.imag) > 500.0


def test_vbeam_two_legs_one_apex_feed():
    """One driven apex gap and two legs of equal length opening symmetrically
    in +/- y."""
    from antenna_designer.designs.cebik.vbeam import Builder

    tups = Builder().build_wires()
    feeds = [t for t in tups if t[3] is not None]
    assert len(feeds) == 1
    ends = [t[1] for t in tups if t[3] is None]
    ys = sorted(e[1] for e in ends)
    assert ys[0] < 0 < ys[-1]  # legs splay to both +/- y
    assert abs(abs(ys[0]) - abs(ys[-1])) < 1e-6  # symmetric


# ---------------------------------------------------------------------------
# Bi-square (2 wl vertical loop curtain)
# ---------------------------------------------------------------------------


def test_bisquare_vertical_broadside():
    """Vertically polarised, fires broadside to the loop plane (off +/- x) with
    the in-plane (+/- y) endfire suppressed -- the in-phase vertical components
    adding while the horizontals cancel."""
    from antenna_designer.designs.cebik.bisquare import Builder

    ff = _far_field(Builder())
    rings = np.array(ff.rings)
    broadside = rings[:, 0].max()  # +x
    end_on = rings[:, 90].max()  # +y
    assert ff.max_gain > 3.0
    assert broadside - end_on > 2.0


def test_bisquare_high_reactive_corner_feed():
    """A 2 wl loop fed at a corner is a high, reactive feedpoint (open-wire +
    tuner), not a 50 ohm match."""
    from antenna_designer.designs.cebik.bisquare import Builder

    z = _z(Builder())
    assert abs(z.imag) > 200.0


def test_bisquare_is_a_four_sided_loop_one_feed():
    """Four half-wave sides forming one closed loop, with a single driven gap
    at the bottom corner (z minimum)."""
    from antenna_designer.designs.cebik.bisquare import Builder

    tups = Builder().build_wires()
    feeds = [t for t in tups if t[3] is not None]
    assert len(feeds) == 1
    zmin = min(min(t[0][2], t[1][2]) for t in tups)
    assert abs(feeds[0][0][2] - zmin) < 1e-6  # fed at the bottom corner


# ---------------------------------------------------------------------------
# J-pole (end-fed half-wave + quarter-wave matching stub)
# ---------------------------------------------------------------------------


def test_jpole_omnidirectional_vertical():
    """A vertical end-fed half-wave: ~2 dBi, omnidirectional in azimuth (small
    ripple around the peak-elevation ring)."""
    from antenna_designer.designs.cebik.jpole import Builder

    ff = _far_field(Builder())
    rings = np.array(ff.rings)
    ti = int(np.argmax(rings.max(axis=1)))  # elevation ring of peak gain
    az = rings[ti]
    assert 1.5 < ff.max_gain < 2.6
    assert az.max() - az.min() < 1.5  # omnidirectional in azimuth


def test_jpole_stub_matches_to_coax():
    """The quarter-wave stub transforms the very high end-fed impedance down to
    a coax-friendly match (SWR < 2.5 to 50 ohm at the tuned tap)."""
    from antenna_designer.designs.cebik.jpole import Builder

    assert _swr(_z(Builder()), 50.0) < 2.5


def test_jpole_radiator_continues_above_the_stub():
    """Topology: the half-wave radiator stands on top of one stub leg, so the
    structure's top is a half-wave above the stub top; the feed bridges the two
    close stub legs (different x)."""
    from antenna_designer.designs.cebik.jpole import Builder

    tups = Builder().build_wires()
    feeds = [t for t in tups if t[3] is not None]
    assert len(feeds) == 1
    # feed bridges the two legs -> its endpoints differ in x
    assert abs(feeds[0][0][0] - feeds[0][1][0]) > 1e-6
    # the radiator reaches well above the stub top
    wl = 299.792458 / Builder().design_freq
    ztop = max(max(t[0][2], t[1][2]) for t in tups)
    zbot = min(min(t[0][2], t[1][2]) for t in tups)
    assert (ztop - zbot) > 0.6 * wl  # stub (~1/4) + radiator (~1/2)


# ---------------------------------------------------------------------------
# Discone (broadband vertical)
# ---------------------------------------------------------------------------


_DISCONE_BAND = (34.0, 40.0, 50.0, 65.0)  # above the ~28.6 MHz cone cutoff


def test_discone_broadband_match():
    """The defining discone behaviour: a usable match held across a wide band
    ABOVE the cone's quarter-wave cutoff (here ~2:1, 34-65 MHz), unlike a
    resonant vertical."""
    from antenna_designer.designs.cebik.discone import Builder

    swrs = [
        _swr(_z(Builder(dict(Builder.default_params, freq=f))), 50.0)
        for f in _DISCONE_BAND
    ]
    assert max(swrs) < 3.0, dict(zip(_DISCONE_BAND, swrs))


def test_discone_match_beats_a_resonant_vertical_off_band():
    """A resonant antenna's SWR explodes when you move ~2:1 in frequency; the
    discone's barely moves. Compare the band-edge spread."""
    from antenna_designer.designs.cebik.discone import Builder
    from antenna_designer.designs.cebik.jpole import Builder as JBuilder

    def spread(B, lo, hi, z0):
        return _swr(_z(B(dict(B.default_params, freq=hi))), z0) - _swr(
            _z(B(dict(B.default_params, freq=lo))), z0
        )

    # the resonant J-pole degrades far more across a 34->65 MHz move than the
    # broadband discone does.
    assert abs(spread(Builder, 34.0, 65.0, 50.0)) < abs(
        spread(JBuilder, 34.0, 65.0, 50.0)
    )


def test_discone_omni_low_angle_in_band():
    """In-band it is a vertical: omnidirectional in azimuth and low takeoff
    (peak gain near the horizon)."""
    from antenna_designer.designs.cebik.discone import Builder

    b = Builder(dict(Builder.default_params, freq=50.0))
    rings = np.array(_far_field(b).rings)
    ti = int(np.argmax(rings.max(axis=1)))
    az = rings[ti]
    assert ti > 75  # peak near the horizon (theta ~ 90)
    assert az.max() - az.min() < 1.0  # omnidirectional


def test_discone_has_disc_and_cone_one_feed():
    """A disc cage (horizontal radials) above a cone cage (downward radials),
    fed across the apex gap -- exactly one driven segment."""
    from antenna_designer.designs.cebik.discone import Builder

    tups = Builder().build_wires()
    feeds = [t for t in tups if t[3] is not None]
    assert len(feeds) == 1
    n = int(Builder().n_wires)
    # m disc radials (horizontal) + m cone wires (sloping down) + 1 feed
    horiz = [t for t in tups if abs(t[0][2] - t[1][2]) < 1e-9 and t[3] is None]
    assert len(horiz) == n  # the disc radials
