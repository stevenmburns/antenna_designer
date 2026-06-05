"""Tests for the pysim-backed SimulationEngine and the flat-wire-to-polyline
geometry translator it sits on top of."""
import numpy as np
import pytest

from antenna_designer.designs.dipole import Builder
from antenna_designer.engines import PyNECEngine, PysimEngine
from antenna_designer.geometry import flat_wires_to_polylines


def test_translator_chains_dipole_into_single_polyline():
    b = Builder()
    out = flat_wires_to_polylines(b.build_wires())

    assert len(out["polylines"]) == 1
    polyline = out["polylines"][0]
    assert polyline.shape == (4, 3), polyline.shape
    assert out["edge_segments"] == [[21, 3, 21]]
    assert out["feed_wire_index"] == 0

    # Feed sits at the geometric centre of the dipole; the dipole spans
    # -length/2 ... +length/2 about x=0, so arclength from the starting
    # end to the feed midpoint is length/2.
    assert out["feed_arclength"] == pytest.approx(b.length / 2.0, rel=1e-9)
    assert out["feed_voltage"] == 1 + 0j


def test_pysim_impedance_in_realistic_range():
    z, = PysimEngine(Builder()).impedance()
    assert z.real > 30 and z.real < 150, f"unrealistic R: {z}"
    # Imaginary part can swing widely with formulation/ground, just sanity-
    # check it stays in a plausible band rather than blowing up.
    assert abs(z.imag) < 200, f"unrealistic X: {z}"


def test_pysim_impedance_sweep_shape_and_monotone_resistance():
    freqs = np.linspace(28.0, 29.0, 5)
    zs = PysimEngine(Builder()).impedance_sweep(freqs)
    assert zs.shape == (5, 1)
    # Driver R rises smoothly across a sub-resonant span for a dipole.
    real = zs[:, 0].real
    assert np.all(np.diff(real) > 0), real


def test_pysim_matches_pynec_in_free_space():
    """Free-space cross-check between the two MoM engines on a dipole.
    Disabling PyNEC's gn_card so both solve the same physical problem
    (no ground, no Fresnel) brings real-part agreement well under 10%
    and reactance close enough to confirm the translator's feed-point
    mapping is correct."""
    b = Builder()
    z_nec, = PyNECEngine(b, ground=None).impedance()
    z_pysim, = PysimEngine(b).impedance()
    real_rel = abs(z_pysim.real - z_nec.real) / abs(z_nec.real)
    assert real_rel < 0.10, f"real parts diverged: nec={z_nec}, pysim={z_pysim}"
    # Reactance offsets between formulations are larger at sub-resonant
    # dipole lengths; absolute, not relative, headroom is the right test.
    assert abs(z_pysim.imag - z_nec.imag) < 20.0, f"reactance diverged: nec={z_nec}, pysim={z_pysim}"


def test_pysim_engine_declares_no_far_field():
    assert PysimEngine.supports_far_field is False
    with pytest.raises(NotImplementedError):
        PysimEngine(Builder()).far_field(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
