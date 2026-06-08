from . import Antenna
from .core import save_or_show
from .engine import SimulationEngine

import numpy as np

import matplotlib.pyplot as plt


def _as_engine(obj):
    """Accept either an AntennaBuilder (legacy path; wrap with the default
    Antenna alias = PyNECEngine) or an already-constructed SimulationEngine
    instance (lets callers pick the backend and any ground/options)."""
    if isinstance(obj, SimulationEngine):
        return obj
    return Antenna(obj)


def _default_name(obj):
    if isinstance(obj, SimulationEngine):
        return type(obj).__name__
    return "Unknown"


# Visual floor for dBi polar plots. Without an explicit rmin matplotlib
# autoscales to data extent, so a constant-radius cut (e.g. an elevation
# slice along a horizontal dipole's broadside direction, gain ≈ const)
# gets smeared across the entire radial range and a 0.02 dBi difference
# between two engines reads as "one curve is at the rim, the other at
# the centre". Pinning the floor to the lowest labelled tick keeps
# constant curves at their actual dBi position.
_DBI_FLOOR = -12


def _init_dbi_polar(ax):
    ax.set_rticks([-12, -6, 0, 6, 12])
    ax.set_rmin(_DBI_FLOOR)


def _finalise_dbi_polar(ax):
    # Let the top expand if data exceeds the highest labelled tick, but
    # never shrink below it — otherwise a low-gain pattern looks identical
    # in shape to a high-gain one because both fill the axis.
    top = max(12, ax.get_ylim()[1])
    ax.set_ylim(_DBI_FLOOR, top)


def get_pattern_rings(builder_or_engine):
    a = _as_engine(builder_or_engine)
    ff = a.far_field(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
    del a
    return ff.rings, ff.max_gain, ff.min_gain, ff.thetas, ff.phis


def get_elevation(a):
    ff = a.far_field(n_theta=90, n_phi=1, del_theta=1, del_phi=360)
    return ff.rings, ff.max_gain, ff.min_gain, ff.thetas, ff.phis


def plot_patterns(
    rings_lst,
    names,
    thetas,
    phis,
    elevation_angle=15,
    fn=None,
    azimuth_f=0,
    azimuth_r=180,
):
    fig, axes = plt.subplots(
        ncols=2, subplot_kw={"projection": "polar"}, figsize=(12, 8)
    )

    _init_dbi_polar(axes[0])

    for nm, rings in zip(names, rings_lst):
        for theta, ring in list(zip(thetas, rings)):
            if abs(theta - (90 - elevation_angle)) < 0.1:
                axes[0].plot(
                    np.deg2rad(phis), ring, marker="", label=f"{(90 - theta):.0f} {nm}"
                )

    _finalise_dbi_polar(axes[0])
    axes[0].legend(loc="lower left")

    n = len(rings_lst[0][0])
    assert (n - 1) % 2 == 0

    if False:
        azimuth_f = 0
        azimuth_r = (n - 1) // 2

        delta_azimuth = 0
        azimuth_f -= delta_azimuth
        azimuth_f %= n - 1

        azimuth_r += delta_azimuth
        azimuth_r %= n - 1

    print(n, azimuth_f, azimuth_r)

    assert 0 <= azimuth_f < n - 1
    assert 0 <= azimuth_r < n - 1

    elevations = [
        list(reversed([ring[azimuth_f] for ring in rings]))
        + [ring[azimuth_r] for ring in rings]
        for rings in rings_lst
    ]
    el_thetas = list(reversed(list(90 - thetas))) + list(90 + thetas)

    _init_dbi_polar(axes[1])

    for elevation in elevations:
        axes[1].plot(np.deg2rad(el_thetas), elevation, marker="")

    _finalise_dbi_polar(axes[1])
    save_or_show(plt, fn)


def compare_patterns(
    builders_or_engines,
    elevation_angle=15,
    fn=None,
    builder_names=None,
    azimuth_f=0,
    azimuth_r=180,
):
    """Plot azimuth + elevation cuts for a sequence of antennas.

    Each item may be either an AntennaBuilder (uses the default PyNEC
    engine) or a pre-constructed SimulationEngine instance — the latter
    is how you pick a non-default backend or ground configuration. Pass
    an explicit `builder_names=[...]` to control legend labels; absent
    that, engine instances get their class name (e.g. "PyNECEngine",
    "PysimEngine") and bare builders fall back to "Unknown" for
    backwards compatibility."""
    if builder_names is None:
        builder_names = [_default_name(b) for b in builders_or_engines]

    rings_lst = []

    for item in builders_or_engines:
        rings, max_gain, min_gain, thetas, phis = get_pattern_rings(item)
        rings_lst.append(rings)

    plot_patterns(
        rings_lst,
        builder_names,
        thetas,
        phis,
        elevation_angle,
        fn,
        azimuth_f,
        azimuth_r,
    )


def pattern(builder_or_engine, elevation_angle=15, fn=None):

    rings, max_gain, min_gain, thetas, phis = get_pattern_rings(builder_or_engine)

    elevation = [ring[0] for ring in rings]

    fig, axes = plt.subplots(ncols=2, subplot_kw={"projection": "polar"})

    _init_dbi_polar(axes[0])

    for theta, ring in list(zip(thetas, rings)):
        if abs(theta - (90 - elevation_angle)) < 0.1:
            axes[0].plot(np.deg2rad(phis), ring, marker="", label=f"{(90 - theta):.0f}")

    _finalise_dbi_polar(axes[0])
    axes[0].legend(loc="lower left")

    n = len(rings[0])
    assert (n - 1) % 2 == 0
    elevation = list(reversed([ring[0] for ring in rings])) + [
        ring[(n - 1) // 2] for ring in rings
    ]
    el_thetas = list(reversed(list(90 - thetas))) + list(90 + thetas)

    _init_dbi_polar(axes[1])

    axes[1].plot(np.deg2rad(el_thetas), elevation, marker="")

    _finalise_dbi_polar(axes[1])
    save_or_show(plt, fn)


def pattern3d(builder_or_engine, fn=None):
    a = _as_engine(builder_or_engine)
    ff = a.far_field(n_theta=30, n_phi=60, del_theta=3, del_phi=6)
    del a

    rhos = [
        [ff.rings[theta_index][phi_index] for theta_index, _ in enumerate(ff.thetas)]
        for phi_index, _ in enumerate(ff.phis)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    Theta, Phi = np.meshgrid(np.deg2rad(ff.thetas), np.deg2rad(ff.phis))
    Rho = 10 ** (np.array(rhos) / 10)

    X = Rho * np.sin(Theta) * np.cos(Phi)
    Y = Rho * np.sin(Theta) * np.sin(Phi)
    Z = Rho * np.cos(Theta)

    ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
    ax.set_aspect("equal")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    save_or_show(plt, fn)
