from . import Antenna
from .core import save_or_show
from .engine import SimulationEngine

import numpy as np

# matplotlib.pyplot is imported lazily inside the plotting functions below
# (plot_patterns / pattern / pattern3d) — it costs ~0.1 s to import and only the
# plotting paths need it, so it stays off `import antennaknobs` and web startup.


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


def _beamwidth_wrapped(angles, gains, peak_idx, threshold):
    """−3 dB-style width through a peak in a *wrapped* cut (azimuth: φ 0→360
    with a duplicated endpoint). Returns the angular width (deg) where the
    trace stays at/above `threshold` either side of the peak, the crossings
    found by linear interpolation between 1° samples. A trace that never drops
    below threshold (omnidirectional) returns the full 360°."""
    g = np.asarray(gains[:-1], float)  # drop the duplicated φ=360 sample
    ang = np.asarray(angles[:-1], float)
    n = len(g)
    if n == 0 or np.all(g >= threshold):
        return 360.0
    pk = int(peak_idx) % n

    def walk(direction):
        dist = 0.0
        i = pk
        for _ in range(n):
            j = (i + direction) % n
            step = ((ang[j] - ang[i]) * direction) % 360.0 or 360.0 / n
            if g[j] < threshold:
                frac = (g[i] - threshold) / (g[i] - g[j])
                return dist + frac * step
            dist += step
            i = j
        return None

    right, left = walk(+1), walk(-1)
    if right is None or left is None:
        return 360.0
    return right + left


def _beamwidth_linear(angles, gains, peak_idx, threshold):
    """−3 dB-style width through a peak in a *non-wrapped* cut (elevation:
    bounded to the 0°–90° hemisphere). Same interpolation as the wrapped case,
    but a side that runs into the array edge without crossing returns the
    distance to the edge, so the result is a lower bound for a lobe that hugs
    the horizon or zenith."""
    g = np.asarray(gains, float)
    ang = np.asarray(angles, float)
    n = len(g)
    if n == 0 or np.all(g >= threshold):
        return abs(float(ang[-1] - ang[0])) if n else 0.0

    def walk(direction):
        dist = 0.0
        i = int(peak_idx)
        while 0 <= i + direction < n:
            j = i + direction
            step = abs(float(ang[j] - ang[i]))
            if g[j] < threshold:
                frac = (g[i] - threshold) / (g[i] - g[j])
                return dist + frac * step
            dist += step
            i = j
        return dist  # ran into the edge — truncated lower bound

    return walk(+1) + walk(-1)


def pattern_metrics(ff, *, beamwidth_db=3.0):
    """Summarise a `FarField` into scalar metrics for comparing antennas.

    Returns a dict with:
      * `peak_gain_dbi`    — maximum gain over the whole pattern (dBi)
      * `takeoff_deg`      — elevation angle of the peak (90 − θ)
      * `azimuth_deg`      — azimuth of the peak
      * `front_to_back_db` — peak minus the gain at the same elevation, 180°
                             away in azimuth
      * `az_beamwidth_deg` — −`beamwidth_db` width through the peak in the
                             azimuth ring at the peak's elevation
      * `el_beamwidth_deg` — −`beamwidth_db` width through the peak in the
                             elevation column at the peak's azimuth (a lower
                             bound when the lobe meets the 0°/90° limit)
    """
    rings = np.asarray(ff.rings, float)
    thetas = np.asarray(ff.thetas, float)
    phis = np.asarray(ff.phis, float)
    ti, pi = np.unravel_index(int(np.argmax(rings)), rings.shape)
    peak = float(rings[ti, pi])
    ring = rings[ti]

    back_az = (float(phis[pi]) + 180.0) % 360.0
    bi = int(np.argmin(np.abs(((phis - back_az + 180.0) % 360.0) - 180.0)))
    front_to_back = peak - float(ring[bi])

    thr = peak - beamwidth_db
    return {
        "peak_gain_dbi": peak,
        "takeoff_deg": float(90.0 - thetas[ti]),
        "azimuth_deg": float(phis[pi]),
        "front_to_back_db": front_to_back,
        "az_beamwidth_deg": _beamwidth_wrapped(phis, ring, pi, thr),
        "el_beamwidth_deg": _beamwidth_linear(90.0 - thetas, rings[:, pi], ti, thr),
    }


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
    import matplotlib.pyplot as plt

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


def _print_metrics_table(names, metrics_lst):
    """Print an aligned metrics table comparing the antennas, so a
    `compare_patterns` run (and the optimize before/after) reports the numbers
    that make the overlaid plot actionable, not just the shapes."""
    cols = [
        ("peak dBi", "peak_gain_dbi", "{:.2f}"),
        ("takeoff°", "takeoff_deg", "{:.0f}"),
        ("F/B dB", "front_to_back_db", "{:.1f}"),
        ("az bw°", "az_beamwidth_deg", "{:.0f}"),
        ("el bw°", "el_beamwidth_deg", "{:.0f}"),
    ]
    name_w = max([len("design")] + [len(str(n)) for n in names])
    header = "design".ljust(name_w) + "  " + "  ".join(h.rjust(8) for h, _, _ in cols)
    print(header)
    print("-" * len(header))
    for nm, m in zip(names, metrics_lst):
        row = str(nm).ljust(name_w)
        for _, key, fmt in cols:
            row += "  " + fmt.format(m[key]).rjust(8)
        print(row)


def compare_patterns(
    builders_or_engines,
    elevation_angle=15,
    fn=None,
    builder_names=None,
    azimuth_f=0,
    azimuth_r=180,
    show_metrics=True,
):
    """Plot azimuth + elevation cuts for a sequence of antennas.

    Each item may be either an AntennaBuilder (uses the default PyNEC
    engine) or a pre-constructed SimulationEngine instance — the latter
    is how you pick a non-default backend or ground configuration. Pass
    an explicit `builder_names=[...]` to control legend labels; absent
    that, engine instances get their class name (e.g. "PyNECEngine",
    "MomwireEngine") and bare builders fall back to "Unknown" for
    backwards compatibility. With `show_metrics` (default) a peak-gain /
    takeoff / F-B / beamwidth table is printed alongside the plot."""
    if builder_names is None:
        builder_names = [_default_name(b) for b in builders_or_engines]

    rings_lst = []
    metrics_lst = []
    thetas = phis = None

    for item in builders_or_engines:
        a = _as_engine(item)
        ff = a.far_field(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
        del a
        rings_lst.append(ff.rings)
        metrics_lst.append(pattern_metrics(ff))
        thetas, phis = ff.thetas, ff.phis

    if show_metrics:
        _print_metrics_table(builder_names, metrics_lst)

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
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt

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
