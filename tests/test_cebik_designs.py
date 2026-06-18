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
