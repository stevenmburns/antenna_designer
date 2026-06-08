"""Bridge antenna_designer's Builder idiom into pysim's web AntennaExample.

Each `designs/<name>.py` exposes a `Builder` class with `default_params`
(a MappingProxyType of physics knobs). We walk that registry, derive a
`ParamSpec` schema from `default_params` (with optional per-design
overrides under the reserved `ui_params` key), and register one
`AntennaExample` per design so the existing pysim web frontend can
drive it without per-design glue.

Reserved keys inside `ui_params`:
  default_view     : "xy" | "yz" | "xz"  — initial 2D projection
  target_z0        : float — reference impedance for SWR (default 50)
  meas_freq_range  : (lo, hi)  — measurement-freq slider span override
  bands            : tuple[BandSpec] — band tabs (default HF amateur set)
  sweep_policy     : (anchor, lo_factor, hi_factor)
  multi_feed       : bool — declare multi-feed response shape
  <param_name>     : dict of {min, max, step, unit, label, precision,
                              kind, sweepable, enum_options}
                     — slider-bounds + metadata overrides for one param.
                     Anything missing falls back to auto-derived defaults.

Everything else in `default_params` becomes a `ParamSpec`. Numeric
defaults become float sliders with auto bounds (±50% around default);
ints become int sliders; bools become checkboxes; complex defaults are
skipped (no UI yet — the request can still override via re/im dict).
"""

from __future__ import annotations

import importlib
import pathlib
import time
from typing import Any

import numpy as np

from antenna_designer.engines.pysim import PysimEngine
from pysim import BSplinePySim, SinusoidalPySim, TriangularPySim

from .examples import register
from .examples._base import (
    DEFAULT_HF_BANDS,
    DEFAULT_SWEEP_POLICY,
    AntennaExample,
    BandSpec,
    ParamSpec,
    SweepPolicy,
)

C_LIGHT = 299_792_458.0

DESIGNS_PKG = "antenna_designer.designs"
DESIGNS_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "antenna_designer" / "designs"
)

_PYSIM_MODELS = {
    "triangular": TriangularPySim,
    "sinusoidal": SinusoidalPySim,
    "bspline": BSplinePySim,
}


# ---------------------------------------------------------------------------
# Schema derivation
# ---------------------------------------------------------------------------


def _strip_ui(params: dict) -> dict:
    """Return a copy of the params dict with the reserved `ui_params` key
    removed — what gets passed into Builder construction."""
    return {k: v for k, v in params.items() if k != "ui_params"}


def _auto_paramspec(name: str, default: Any, override: dict | None) -> ParamSpec | None:
    """Build a ParamSpec from a default value plus optional UI overrides.

    Returns None when the value type has no UI representation (complex,
    string-non-enum, etc.) and no override was supplied — the param is
    still settable via the API, it just doesn't appear in the UI.
    """
    override = dict(override or {})
    label = override.pop("label", name)
    unit = override.pop("unit", None)
    precision = int(override.pop("precision", 3))
    sweepable = bool(override.pop("sweepable", name == "freq"))

    if isinstance(default, bool):
        kind = override.pop("kind", "bool")
        return ParamSpec(
            name=name,
            label=label,
            default=default,
            kind=kind,
            unit=unit,
            precision=precision,
        )

    if isinstance(default, (int, float)) and not isinstance(default, bool):
        is_int = isinstance(default, int) and override.get("kind") != "float"
        kind = override.pop("kind", "int" if is_int else "float")
        d = float(default)
        # Auto bounds: a generous ±50% window with 100 steps. For an int
        # default of 0 the multiplicative window collapses, so fall back
        # to a small absolute range.
        if d == 0.0:
            lo, hi, step = -1.0, 1.0, 0.1
        else:
            lo = d * 0.5 if d > 0 else d * 1.5
            hi = d * 1.5 if d > 0 else d * 0.5
            step = max((hi - lo) / 100.0, 1e-6)
        if kind == "int":
            lo = float(int(round(lo)))
            hi = float(int(round(hi)))
            step = 1.0
            precision = 0
        # `design_freq` is the geometry-sizing frequency for
        # freq_based.* designs (wavelength = c / design_freq, then
        # dimensions are wavelength × factors). Wire it into the
        # global designFreq state on the frontend so the slider
        # actually retunes the geometry AND the meas-freq slider
        # follows when linkMeas is on. Top-level designs don't have a
        # design_freq param — their geometry is hand-tuned in absolute
        # meters and the measurement freq slider (at the top of the
        # UI) is the only thing that needs to move per solve.
        if name == "design_freq":
            unit = unit or "MHz"
            override["linked_to_design_freq"] = True  # keep around
        spec_kwargs = dict(
            name=name,
            label=label,
            default=int(d) if kind == "int" else d,
            kind=kind,
            min=float(override.pop("min", lo)),
            max=float(override.pop("max", hi)),
            step=float(override.pop("step", step)),
            precision=precision,
            unit=unit,
            sweepable=sweepable,
        )
        if "linked_to_design_freq" in override:
            spec_kwargs["linked_to_design_freq"] = bool(
                override.pop("linked_to_design_freq")
            )
        return ParamSpec(**spec_kwargs)

    if isinstance(default, str):
        opts = override.pop("enum_options", None)
        if opts is None:
            return None
        return ParamSpec(
            name=name,
            label=label,
            default=default,
            kind="enum",
            enum_options=tuple(opts),
            precision=precision,
            unit=unit,
        )

    # complex, None, or anything exotic — skip the auto-UI; the request
    # body can still override via {"re": ..., "im": ...}.
    return None


def _derive_schema(default_params: dict) -> tuple[ParamSpec, ...]:
    ui = dict(default_params.get("ui_params") or {})
    specs: list[ParamSpec] = []
    for key, default in default_params.items():
        if key == "ui_params":
            continue
        # `freq` is measurement frequency only — driven by the dedicated
        # meas-freq slider at the top of the UI, never by a schema
        # slider. The Builder's default_params['freq'] value is still
        # used as the initial measurement freq when the example loads;
        # the adapter just doesn't expose a redundant slider for it.
        if key == "freq":
            continue
        override = ui.get(key)
        if override is not None and not isinstance(override, dict):
            # Reserved scalar (e.g. `target_z0`) — not a per-param spec.
            continue
        spec = _auto_paramspec(key, default, override)
        if spec is not None:
            specs.append(spec)
    return tuple(specs)


# ---------------------------------------------------------------------------
# Builder construction from a request dict
# ---------------------------------------------------------------------------


def _rehydrate_param(default_value: Any, raw: Any) -> Any:
    if isinstance(default_value, complex) and isinstance(raw, dict):
        return complex(float(raw.get("re", 0.0)), float(raw.get("im", 0.0)))
    if isinstance(default_value, bool):
        return bool(raw)
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw)
    if isinstance(default_value, float):
        return float(raw)
    return raw


def _build_builder(cls, req: dict):
    """Construct a Builder from default_params overlaid with request fields.

    The pysim frontend assembles its solve request by Object.assign'ing
    every live slider value as a *top-level* key on the request dict
    (App.tsx:buildRequest), so we read each Builder param off the request
    directly. A nested `params` dict is also accepted as a fallback for
    other clients.
    """
    base = _strip_ui(dict(cls.default_params))
    nested = req.get("params") or {}
    for k in list(base.keys()):
        if k in req:
            base[k] = _rehydrate_param(base[k], req[k])
        elif k in nested:
            base[k] = _rehydrate_param(base[k], nested[k])
    return cls(params=base)


def _ground_for_engine(req: dict, ground_z: float):
    ground_on = bool(req.get("ground", False))
    if not ground_on:
        return None
    # Map the frontend's ground knobs to the engine's ground spec. The
    # pysim frontend currently sends ground=True with an implicit PEC;
    # finite ground specs can be wired through later via dedicated UI.
    return "pec"


def _make_pysim_engine(req: dict, builder):
    model = req.get("pysim_model", "triangular")
    solver_cls = _PYSIM_MODELS.get(model, TriangularPySim)
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = _ground_for_engine(req, 0.0)
    solver_kwargs = req.get("model_options") or None
    return PysimEngine(
        builder,
        solver=solver_cls,
        wire_radius=wire_radius,
        solver_kwargs=solver_kwargs,
        ground=ground,
    )


# ---------------------------------------------------------------------------
# Response packing
# ---------------------------------------------------------------------------


# Frontend Fresnel reflection treats this as the real part of the
# complex permittivity. For PEC the reflection coefficient ρ_h → −1 as
# eps_r → ∞; 1e10 is large enough to be numerically indistinguishable
# while staying away from float overflow. Matches pysim/web/server.py.
_PEC_GROUND_EPS_R = 1.0e10
_PEC_GROUND_SIGMA = 0.0


def _pack_wires(currents) -> list[dict]:
    return [
        {
            "label": f"wire{idx}",
            "knot_positions": w.knot_positions.tolist(),
            "knot_currents_re": w.knot_currents.real.tolist(),
            "knot_currents_im": w.knot_currents.imag.tolist(),
        }
        for idx, w in enumerate(currents)
    ]


def _feed_indices(engine, currents) -> tuple[int, int]:
    """Pick a (wire, knot) for the feed marker.

    PysimEngine exposes `_feeds = [(polyline_idx, arclength, voltage)]`
    post-translator. Map the first feed's polyline to the WireCurrents
    list (they're 1:1 in index order) and place the marker on the knot
    closest to that arclength along the polyline.
    """
    feeds = getattr(engine, "_feeds", None) or []
    if not feeds:
        return 0, 0
    pl_idx, arclen, _v = feeds[0]
    if pl_idx >= len(currents):
        return 0, 0
    knots = currents[pl_idx].knot_positions
    if knots.shape[0] < 2:
        return pl_idx, 0
    # Cumulative arclength along the polyline.
    deltas = np.linalg.norm(np.diff(knots, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(deltas)])
    j = int(np.argmin(np.abs(cum - float(arclen))))
    return int(pl_idx), j


# ---------------------------------------------------------------------------
# Example factory
# ---------------------------------------------------------------------------


def _ui_scalar(default_params: dict, key: str, default):
    ui = default_params.get("ui_params") or {}
    if key in ui and not isinstance(ui[key], dict):
        return ui[key]
    return default


def _auto_default_view(cls) -> str:
    """Pick a 2D projection from the spans of the antenna's wires.

    Rule: if x_span is small (the antenna lies in the y-z plane —
    typical for dipoles, V's, loops, fan/bowtie variants), default to
    `yz`. Otherwise return the plane of the two largest spans (xy / yz
    / xz). The 0.5 m threshold catches feed-gap micro-offsets like
    fan_dipole's 0.22 m without flipping to xy.

    Hand-overridden via ui_params['default_view']; designs whose axis
    layout doesn't match this rule (vertical, moxonarray) supply the
    explicit value.
    """
    try:
        b = cls()
        pts = []
        for p0, p1, _n, _e in b.build_wires():
            pts.append(p0)
            pts.append(p1)
        a = np.asarray(pts, dtype=float)
    except Exception:
        return "xy"
    sx = float(a[:, 0].max() - a[:, 0].min())
    sy = float(a[:, 1].max() - a[:, 1].min())
    sz = float(a[:, 2].max() - a[:, 2].min())
    if sx < 0.5:
        return "yz"
    spans = sorted([("x", sx), ("y", sy), ("z", sz)], key=lambda t: t[1], reverse=True)
    return "".join(sorted(s[0] for s in spans[:2]))


def _make_example(name: str, cls) -> AntennaExample:
    dp = dict(cls.default_params)
    ui = dict(dp.get("ui_params") or {})

    default_view = _ui_scalar(dp, "default_view", _auto_default_view(cls))
    target_z0 = float(_ui_scalar(dp, "target_z0", 50.0))  # noqa: F841 — surfaced later
    multi_feed = bool(_ui_scalar(dp, "multi_feed", False))
    meas_range = (
        ui.get("meas_freq_range")
        if not isinstance(ui.get("meas_freq_range"), dict)
        else None
    )
    bands_override = ui.get("bands") if not isinstance(ui.get("bands"), dict) else None
    sweep_pol_raw = ui.get("sweep_policy")
    if isinstance(sweep_pol_raw, (tuple, list)) and len(sweep_pol_raw) == 3:
        sweep_policy = SweepPolicy(
            anchor=str(sweep_pol_raw[0]),
            lo_factor=float(sweep_pol_raw[1]),
            hi_factor=float(sweep_pol_raw[2]),
        )
    else:
        sweep_policy = DEFAULT_SWEEP_POLICY

    # Band tabs default to the HF amateur set in canonical order. The
    # frontend snaps to whichever band contains the design's native
    # `freq` (looked up from the param schema's freq default) — see
    # the useEffect on currentExample in App.tsx. Designs can still
    # override via ui_params['bands'].
    if bands_override is not None:
        bands = tuple(BandSpec(*b) for b in bands_override)
    else:
        bands = DEFAULT_HF_BANDS

    param_schema = _derive_schema(dp)

    def pysim_solve(req: dict) -> dict:
        design_freq = float(req.get("design_freq_mhz", dp.get("freq", 14.0)))
        meas_freq = float(req.get("measurement_freq_mhz", design_freq))
        builder = _build_builder(cls, req)
        builder.freq = meas_freq
        eng = _make_pysim_engine(req, builder)
        t0 = time.perf_counter()
        zs = eng.impedance()
        currents = eng.current_distribution()
        solve_ms = (time.perf_counter() - t0) * 1e3
        feed_wire_idx, feed_knot_idx = _feed_indices(eng, currents)
        z_primary = zs[0] if zs else complex(0.0, 0.0)
        out = {
            "geometry": name,
            "wires": _pack_wires(currents),
            "feed_wire_index": feed_wire_idx,
            "feed_knot_index": feed_knot_idx,
            "z_in_re": float(z_primary.real),
            "z_in_im": float(z_primary.imag),
            "design_freq_mhz": design_freq,
            "measurement_freq_mhz": meas_freq,
            "lambda_design_m": C_LIGHT / (design_freq * 1e6),
            "solve_ms": solve_ms,
            "ground": bool(req.get("ground", False)),
            "height_m": 0.0,
            "ground_eps_r": _PEC_GROUND_EPS_R,
            "ground_sigma": _PEC_GROUND_SIGMA,
        }
        if multi_feed and len(zs) > 1:
            out["feeds"] = [{"z_re": float(z.real), "z_im": float(z.imag)} for z in zs]
        return out

    def pysim_sweep(req: dict, freqs_mhz: list[float]):
        builder = _build_builder(cls, req)
        # PysimEngine reads builder.freq only for the initial wavelength
        # passed to _make_solver — impedance_sweep overrides k per point.
        builder.freq = float(freqs_mhz[0]) if freqs_mhz else float(builder.freq)
        eng = _make_pysim_engine(req, builder)
        zs = np.asarray(eng.impedance_sweep(list(freqs_mhz)))
        # PysimEngine.impedance_sweep returns (n_freqs, n_feeds).
        primary = zs[:, 0]
        re = primary.real.tolist()
        im = primary.imag.tolist()
        if multi_feed and zs.shape[1] > 1:
            feeds_re = zs.real.tolist()  # (n_freqs, n_feeds) list of lists
            feeds_im = zs.imag.tolist()
            return re, im, feeds_re, feeds_im
        return re, im

    return AntennaExample(
        name=name,
        label=name.replace("_", " "),
        pysim_solve=pysim_solve,
        pysim_sweep=pysim_sweep,
        multi_feed=multi_feed,
        param_schema=param_schema,
        bands=bands,
        meas_freq_range_mhz=tuple(meas_range) if meas_range else None,
        sweep_policy=sweep_policy,
        default_view=default_view,
        default_freq_mhz=float(dp["freq"]) if "freq" in dp else None,
    )


# ---------------------------------------------------------------------------
# Registration entrypoint
# ---------------------------------------------------------------------------


def list_designs() -> list[str]:
    """Discover every Builder file under designs/.

    Top-level files register under their bare stem (`invvee`). Files
    inside a subpackage (today only `freq_based/`) register under the
    dotted path the user sees in the UI (`freq_based.invvee`) — same
    convention as the Python import path, minus the leading
    `antenna_designer.designs.`. The dotted name is what `register_all`
    feeds back to importlib too.
    """
    names: list[str] = []
    for p in sorted(DESIGNS_DIR.glob("*.py")):
        if p.stem.startswith("_"):
            continue
        names.append(p.stem)
    for sub in sorted(d for d in DESIGNS_DIR.iterdir() if d.is_dir()):
        if sub.name.startswith("_") or sub.name == "__pycache__":
            continue
        for p in sorted(sub.glob("*.py")):
            if p.stem.startswith("_"):
                continue
            names.append(f"{sub.name}.{p.stem}")
    return names


def register_all() -> list[str]:
    """Walk designs/ and register one AntennaExample per Builder class.

    Returns the list of design names that registered successfully. Any
    individual failure is swallowed and logged (a single broken design
    must not take down the whole web UI).
    """
    registered: list[str] = []
    for name in list_designs():
        try:
            mod = importlib.import_module(f"{DESIGNS_PKG}.{name}")
        except Exception as exc:
            print(f"[adapter] skip {name}: import error: {exc!r}")
            continue
        cls = getattr(mod, "Builder", None)
        if cls is None:
            continue
        try:
            cls()  # smoke-test that default_params constructs cleanly
            register(_make_example(name, cls))
            registered.append(name)
        except Exception as exc:
            print(f"[adapter] skip {name}: {exc!r}")
    return registered
