"""Tests for the explicit-feedpoint Sterba variant (wire.sterba_driven).

The design deletes the 8 interior risers and exposes the 4 ports of each
junction as independent feedpoints. These tests check the *mechanism* — that
placing feedpoints at those ports builds and solves — and the linear-algebra
contract the experiment relies on: voltages V = Y⁻¹·I_target reproduce any
target port currents. (Whether that recovers the curtain's gain is a physics
question the driver script measures, not something pinned here.)
"""

import numpy as np
import pytest

from antennaknobs.designs.wire import sterba_driven
from antennaknobs.engines import MomwireEngine

FF_KW = dict(n_theta=90, n_phi=360, del_theta=1, del_phi=1)


def _builder(active, feed_voltages=None):
    p = dict(sterba_driven.Builder.default_params)
    p["active_junctions"] = active
    p["feed_voltages"] = feed_voltages
    return sterba_driven.Builder(p)


@pytest.mark.parametrize("active", [[2], [2, 3], [1, 2, 3, 4]])
def test_named_edges_match_network_ports(active):
    """Every network PortAtEdge must be backed by a named edge in build_wires
    (the engine asserts this); and there are exactly 4 ports per junction."""
    b = _builder(active)
    named = {t[4] for t in b.build_wires() if len(t) >= 5 and t[4]}
    ports = set(b.build_network().ports)
    assert named == ports
    assert len(ports) == 4 * len(active)


def test_single_junction_solves_to_finite_z_and_gain():
    """The user's 'can it be done': one vertical pair (4 feedpoints) builds,
    solves to finite driving-point impedances, and yields a finite far field."""
    eng = MomwireEngine(_builder([2]), ground=None)
    zs = eng.impedance()
    assert len(zs) == 4
    assert all(np.isfinite(z.real) and np.isfinite(z.imag) for z in zs)
    ff = eng.far_field(**FF_KW)
    assert np.isfinite(ff.max_gain)


def test_current_match_round_trip():
    """V = Y⁻¹·I_target reproduces the target port currents (the contract the
    experiment uses to inject the reference currents). Well-conditioned, so
    the round-trip is exact to solver precision."""
    b = _builder([1, 2, 3, 4])
    eng = MomwireEngine(b, ground=None)
    wl = 299.792458 / b.design_freq
    Y = eng._compute_y_matrix(wl)

    rng = np.random.default_rng(0)
    n = Y.shape[0]
    I_target = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    V = np.linalg.solve(Y, I_target)

    assert np.linalg.cond(Y) < 1e3  # feedpoints are independent, not degenerate
    assert np.allclose(Y @ V, I_target, atol=1e-10)


def test_applied_voltages_realize_the_target_port_currents():
    """Feeding the solved voltages back through the design reproduces the
    target port currents in the actual network reduction (not just the bare Y
    algebra). With I_target = 1 A at every port, the driving-point Z = V/I
    returned by impedance() must equal the applied voltages (since I = 1)."""
    b0 = _builder([2])
    eng0 = MomwireEngine(b0, ground=None)
    wl = 299.792458 / b0.design_freq
    Y = eng0._compute_y_matrix(wl)
    pti = eng0._reducer.port_to_idx

    I_target = np.ones(Y.shape[0], dtype=np.complex128)  # 1 A into every port
    V = np.linalg.solve(Y, I_target)

    fv = {nm: complex(V[i]) for nm, i in pti.items()}
    eng = MomwireEngine(_builder([2], feed_voltages=fv), ground=None)
    zs = eng.impedance()

    # impedance() returns V/I per driven port in source order (j, then At/Bt/
    # Ab/Bb); I = 1 A, so Z must equal the applied voltage at each port.
    order = [f"j2_{t}" for t in ("At", "Bt", "Ab", "Bb")]
    z_expected = [V[pti[nm]] for nm in order]
    assert np.allclose(zs, z_expected, rtol=1e-6, atol=1e-9)
