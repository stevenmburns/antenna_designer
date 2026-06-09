"""Unit tests for web/examples/_feedline.py.

`daisy_chain_z_in` is pure math (z_oc + jumper β·L → combined Z_in), so
every test here builds a synthetic Z_oc matrix and asserts the closed-
form answer. No solver, no FastAPI — each test is microseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from web.examples._feedline import daisy_chain_z_in

C_LIGHT = 299_792_458.0


def test_single_port_returns_z_oc_directly():
    # N=1 has no jumpers; Z_in equals the bare antenna impedance.
    z_oc = np.array([[73.0 + 42.5j]], dtype=complex)
    assert daisy_chain_z_in(z_oc, [], freq_mhz=14.3) == complex(z_oc[0, 0])


def test_non_square_z_oc_raises():
    z_oc = np.zeros((2, 3), dtype=complex)
    with pytest.raises(ValueError, match="square"):
        daisy_chain_z_in(z_oc, [0.0], freq_mhz=14.3)


def test_jumper_count_mismatch_raises():
    z_oc = np.eye(3, dtype=complex)
    with pytest.raises(ValueError, match="jumper lengths"):
        daisy_chain_z_in(z_oc, [1.0], freq_mhz=14.3)


def test_very_short_jumpers_collapse_to_parallel_combination():
    # As jumper length → 0, the cot/csc terms dominate and force all
    # node voltages to be equal; the chain collapses to a single node
    # whose driving-point Z is the (1,1) entry of inv(sum of Y across
    # the ports). Equivalent to shorting all ports together.
    z_oc = np.diag([50.0 + 0j, 100.0 + 0j])
    z = daisy_chain_z_in(z_oc, [1e-9], freq_mhz=14.3)
    # 50 ‖ 100 = 33.33 Ω; the short-jumper limit must be inside ±0.1 Ω.
    assert abs(z.real - (50 * 100) / 150) < 0.1
    assert abs(z.imag) < 0.1


def test_quarter_wavelength_jumper_does_not_blow_up():
    # The implementation nudges β·L = π by 1e-9 to dodge the cot/csc
    # singularity. Confirm the result stays finite and bounded — the
    # actual value at the singularity is a moving target, but it must
    # not be NaN/Inf.
    f_mhz = 14.3
    wavelength = C_LIGHT / (f_mhz * 1e6)
    qw = wavelength / 4
    z_oc = np.array([[60.0 + 10.0j, 0.0], [0.0, 75.0 - 5.0j]], dtype=complex)
    z = daisy_chain_z_in(z_oc, [qw], freq_mhz=f_mhz)
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert abs(z) < 1e6  # generous; just needs to be bounded


def test_z0_override_changes_transformation():
    # Same antenna, same jumper, different z0 → different Z_in. Sanity
    # check that the z0 kwarg actually feeds into the math.
    z_oc = np.diag([50.0 + 0j, 50.0 + 0j])
    f_mhz = 14.3
    wavelength = C_LIGHT / (f_mhz * 1e6)
    # Eighth-wave jumper to keep the cot/csc terms finite + non-trivial.
    eighth = wavelength / 8
    z50 = daisy_chain_z_in(z_oc, [eighth], freq_mhz=f_mhz, z0_ohms=50.0)
    z75 = daisy_chain_z_in(z_oc, [eighth], freq_mhz=f_mhz, z0_ohms=75.0)
    assert z50 != z75
