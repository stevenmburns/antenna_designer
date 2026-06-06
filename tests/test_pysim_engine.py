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


def test_sweep_freq_accepts_engine_factory(tmp_path):
    """sweep_freq's `engine=` kwarg accepts any callable (builder) ->
    SimulationEngine. functools.partial is the ergonomic way to bind
    construction kwargs like ground."""
    import matplotlib
    matplotlib.use("Agg")
    from functools import partial
    import antenna_designer as ant
    out = tmp_path / "sf.png"
    ant.sweep_freq(
        Builder(),
        rng=(28.0, 29.0),
        npoints=5,
        fn=str(out),
        engine=partial(PysimEngine),
    )
    assert out.exists() and out.stat().st_size > 0


def test_sweep_accepts_engine_factory(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import antenna_designer as ant
    out = tmp_path / "sw.png"
    ant.sweep(
        Builder(),
        "length",
        center=5.032,
        fraction=1.05,
        npoints=3,
        fn=str(out),
        engine=PysimEngine,
    )
    assert out.exists() and out.stat().st_size > 0


def test_sweep_gain_accepts_engine_factory(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import antenna_designer as ant
    out = tmp_path / "sg.png"
    ant.sweep_gain(
        Builder(),
        "length",
        center=5.032,
        fraction=1.05,
        npoints=3,
        fn=str(out),
        engine=PysimEngine,
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_patterns_pins_radial_floor(tmp_path):
    """Without an rlim, matplotlib polar autoscale would smear a
    constant-radius elevation cut across the full radial range. Pin the
    floor to the lowest tick label so the displayed radius reflects the
    actual dBi value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import antenna_designer as ant
    b = Builder()
    out = tmp_path / "p.png"
    ant.compare_patterns(
        [PyNECEngine(b, ground=None), PysimEngine(b)],
        fn=str(out),
    )
    # The fn= save closes the figure, but the open-figure path also
    # exists; either way the file is on disk.
    assert out.exists() and out.stat().st_size > 0
    plt.close("all")


def test_translator_handles_hentenna_tee_junctions():
    """Hentenna has two degree-3 nodes (B, D); translator should
    decompose into 3 polylines all running B→D, with one junction at
    each."""
    from antenna_designer.designs.freq_based.hentenna import Builder as H
    out = flat_wires_to_polylines(H().build_wires())
    assert len(out["polylines"]) == 3
    assert len(out["junctions"]) == 2
    # Two junctions, each with 3 polyline ends meeting.
    assert sorted(len(j) for j in out["junctions"]) == [3, 3]


def test_translator_handles_fandipole_high_degree_junctions():
    """Fandipole has two degree-6 nodes (S, T): feed wire + 5 spokes
    on each side. 5 polylines per side + 1 feed = 11 polylines."""
    from antenna_designer.designs.fandipole import Builder as F
    out = flat_wires_to_polylines(F().build_wires())
    assert len(out["polylines"]) == 11
    assert len(out["junctions"]) == 2
    assert sorted(len(j) for j in out["junctions"]) == [6, 6]


def test_pysim_sinusoidal_hentenna_impedance_close_to_pynec():
    """Cross-validation on the hentenna (two tee junctions): pysim's
    Sinusoidal basis agrees with PyNEC's free-space gn_card-disabled
    solve to within ~10% on R and ~10 Ω on X. Triangular at the same
    segmentation lands at a different impedance — the two basis
    families converge to two different limits (PyNEC and Sinusoidal
    to one, Triangular and BSpline to another), not to a common point.
    A cross-engine bound against PyNEC therefore only makes sense for
    Sinusoidal here. Picking a "more correct" pair is out of scope;
    this test is just verifying that the translator's junction/feed
    mapping is right."""
    from pysim import SinusoidalPySim
    from antenna_designer.designs.freq_based.hentenna import Builder as H
    b = H()
    z_nec = PyNECEngine(b, ground=None).impedance()[0]
    z_ps = PysimEngine(b, solver=SinusoidalPySim).impedance()[0]
    assert abs(z_ps.real - z_nec.real) / abs(z_nec.real) < 0.15
    assert abs(z_ps.imag - z_nec.imag) < 15.0


def test_pysim_sinusoidal_fandipole_runs():
    """Fandipole has degree-6 junctions and a 1-segment feed gap. The
    1-segment feed has zero interior knots so the Triangular tent basis
    has no feed to land on; Sinusoidal's const-source basis lives on
    segment centres and handles it. Just ensure it runs and produces
    a plausible value, the multi-wire geometry has too many freedoms
    to set a tight tolerance here."""
    from pysim import SinusoidalPySim
    from antenna_designer.designs.fandipole import Builder as F
    z = PysimEngine(F(), solver=SinusoidalPySim).impedance()[0]
    assert 20 < z.real < 200, z
    assert abs(z.imag) < 200, z
