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


def test_pysim_engine_declares_far_field_support():
    assert PysimEngine.supports_far_field is True


def test_pysim_far_field_shape_matches_pynec():
    """The FarField shape (rings dims, thetas/phis arrays) has to match
    PyNEC's so plot_patterns, compare_patterns etc. work for both."""
    b = Builder()
    ff_nec = PyNECEngine(b, ground=None).far_field(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
    ff_ps = PysimEngine(b).far_field(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
    assert np.array_equal(ff_nec.thetas, ff_ps.thetas)
    assert np.array_equal(ff_nec.phis, ff_ps.phis)
    assert len(ff_ps.rings) == 90
    assert len(ff_ps.rings[0]) == 361


def test_pysim_free_space_directivity_matches_pynec():
    """Free-space dipole peak directivity — same physical problem under
    two independent MoM solvers. 0.1 dBi headroom is generous for what
    is, on the dipole, sub-0.02 dBi agreement in practice."""
    b = Builder()
    ff_nec = PyNECEngine(b, ground=None).far_field(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
    ff_ps = PysimEngine(b).far_field(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
    assert abs(ff_ps.max_gain - ff_nec.max_gain) < 0.1, (ff_nec.max_gain, ff_ps.max_gain)


def test_pysim_pec_ground_directivity_matches_pynec():
    """PEC ground via image method on both sides. Tight agreement
    expected since the physics is identical."""
    b = Builder()
    ff_nec = PyNECEngine(b, ground="pec").far_field(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
    ff_ps = PysimEngine(b, ground="pec").far_field(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
    assert abs(ff_ps.max_gain - ff_nec.max_gain) < 0.1, (ff_nec.max_gain, ff_ps.max_gain)


def test_pysim_finite_ground_returns_sane_values():
    """Finite ground in PysimEngine is PEC-image-plus-Fresnel post-
    processing; PyNEC's gn_card(0,...) uses a more sophisticated
    Sommerfeld/Norton model. The two diverge by ~1.5 dBi on a 10m
    dipole over (eps_r=10, sigma=0.002) ground. Don't claim equality;
    just sanity-check the output."""
    b = Builder()
    ff = PysimEngine(b, ground=("finite", 10.0, 0.002)).far_field(
        n_theta=90, n_phi=360, del_theta=1, del_phi=1
    )
    assert 0.0 < ff.max_gain < 15.0, ff.max_gain
    assert ff.min_gain < ff.max_gain
    assert np.all(np.isfinite([ff.max_gain, ff.min_gain]))


def test_compare_patterns_accepts_engine_instances(tmp_path):
    """End-to-end: compare_patterns with a mix of pre-built engines
    (so the caller picks ground / backend per item) should run to
    completion and produce a non-empty PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import antenna_designer as ant
    b = Builder()
    out = tmp_path / "cmp.png"
    ant.compare_patterns(
        [PyNECEngine(b, ground=None), PysimEngine(b)],
        fn=str(out),
        builder_names=["pynec-free", "pysim-free"],
    )
    assert out.exists() and out.stat().st_size > 0


def test_compare_patterns_backwards_compatible_with_bare_builders(tmp_path):
    """Passing AntennaBuilder instances (the historical API) must keep
    working — they get wrapped with the default Antenna alias."""
    import matplotlib
    matplotlib.use("Agg")
    import antenna_designer as ant
    out = tmp_path / "cmp.png"
    ant.compare_patterns([Builder(), Builder()], fn=str(out))
    assert out.exists() and out.stat().st_size > 0
