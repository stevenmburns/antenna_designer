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

from antenna_designer.builder import (
    Array1x2Builder,
    Array1x4Builder,
    Array1x4GroupedBuilder,
    Array2x2Builder,
    Array2x4Builder,
)
from antenna_designer.engines.pynec import PyNECEngine
from antenna_designer.engines.pysim import PysimEngine
from pysim import BSplinePySim, SinusoidalPySim, TriangularPySim

from .examples import register
from .examples._base import (
    DEFAULT_HF_BANDS,
    DEFAULT_SWEEP_POLICY,
    AntennaExample,
    BandSpec,
    ParamGroupSpec,
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
        # Phase params (phase_lr, phase_tb, ...) are degrees, converted
        # to a phasor by the array builders via exp(j π · phase / 180).
        # ±180° covers the full unit circle; signed range puts the
        # zero-phase reference at slider centre with positive = lead,
        # negative = lag. The auto-derived (-1, 1) fallback for
        # default=0 would otherwise give a useless 2° span.
        if name.startswith("phase_"):
            lo, hi, step = -180.0, 180.0, 1.0
            unit = unit or "°"
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
        if "link_meas_freq_to_param" in override:
            spec_kwargs["link_meas_freq_to_param"] = str(
                override.pop("link_meas_freq_to_param")
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


def _group_spec_from_default(
    name: str,
    default_value: tuple | list,
    ui_override: dict,
    all_default_params: dict,
) -> ParamGroupSpec | None:
    """Build a ParamGroupSpec from a tuple/list-of-dicts default value.

    The default value's length seeds default_overrides for the group's
    instances; the inner ParamSpecs come from auto-deriving each key of
    the first instance dict (with optional per-leaf overrides supplied
    under the same ui_override dict, keyed by leaf name).

    `ui_override` is the dict stored under `ui_params[<group_name>]`.
    Recognised keys: label_template, repeat_count, max_repeats,
    link_meas_freq_to_param, plus any leaf-name → override-dict pairs.
    Falls back to sensible defaults when missing.
    """
    if not default_value or not all(isinstance(d, dict) for d in default_value):
        return None
    template = default_value[0]
    if not template:
        return None

    repeat_count = ui_override.get("repeat_count")
    if repeat_count is None:
        # Heuristic: prefer n_<name> (n_bands for bands), then n_<singular>.
        for cand in (f"n_{name}", f"n_{name.rstrip('s')}"):
            if cand in all_default_params:
                repeat_count = cand
                break
    if not isinstance(repeat_count, str):
        # No count param → can't render a repeating group.
        return None

    max_repeats = int(ui_override.get("max_repeats", len(default_value)))
    label_template = str(ui_override.get("label_template", f"{name} {{i}}"))
    link = ui_override.get("link_meas_freq_to_param")

    inner_params: list[ParamSpec] = []
    for leaf_name, leaf_default in template.items():
        leaf_override = ui_override.get(leaf_name)
        if leaf_override is None or not isinstance(leaf_override, dict):
            leaf_override = {}
        spec = _auto_paramspec(leaf_name, leaf_default, dict(leaf_override))
        if spec is not None:
            inner_params.append(spec)
    if not inner_params:
        return None

    default_overrides = tuple(dict(d) for d in default_value)

    return ParamGroupSpec(
        name=name,
        label_template=label_template,
        repeat_count=repeat_count,
        max_repeats=max_repeats,
        params=tuple(inner_params),
        default_overrides=default_overrides,
        link_meas_freq_to_param=str(link) if isinstance(link, str) else None,
    )


def _derive_schema(default_params: dict) -> tuple:
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
        #
        # `design_freq` is the geometry-sizing frequency for
        # freq_based.* designs, driven by the "design freq" band-tab
        # row + slider in the UI (which sends design_freq_mhz on the
        # request). Skipping it here too prevents the auto-derived
        # schema slider from duplicating that control.
        if key in ("freq", "design_freq"):
            continue
        # Repeating-group default: tuple/list of dicts → ParamGroupSpec.
        # The ui_params override (if any) carries the group-level
        # config (label_template, repeat_count, max_repeats,
        # link_meas_freq_to_param) plus per-leaf override dicts.
        if (
            isinstance(default, (tuple, list))
            and default
            and all(isinstance(x, dict) for x in default)
        ):
            group_override = ui.get(key)
            if not isinstance(group_override, dict):
                group_override = {}
            group_spec = _group_spec_from_default(
                key, default, group_override, default_params
            )
            if group_spec is not None:
                specs.append(group_spec)
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
    # Seed from the named variant (e.g. `opt_params`, `z50_params`).
    # Unrecognised / absent → fall back to default_params.
    base = _strip_ui(_variant_params(cls, req.get("variant")))
    nested = req.get("params") or {}
    for k in list(base.keys()):
        if k in req:
            base[k] = _rehydrate_param(base[k], req[k])
        elif k in nested:
            base[k] = _rehydrate_param(base[k], nested[k])
    builder = cls(params=base)
    # n_per_wire drives the per-Builder nominal_nsegs (the convergence
    # sweep at /converge overrides this value per N). Each generator
    # decides which per-edge segment counts scale with it and which stay
    # fixed (feed gaps). See AntennaBuilder.FRAMEWORK_PARAMS.
    n_per_wire = req.get("n_per_wire")
    if n_per_wire is not None:
        builder.nominal_nsegs = int(n_per_wire)
    return builder


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


def _make_pynec_engine(req: dict, builder):
    # PyNECEngine accepts the same ground-spec vocabulary as PysimEngine
    # — None / "pec" / ("finite", eps_r, sigma) — but defaults to a
    # finite ground rather than free space if you pass nothing. Match
    # PysimEngine's behaviour: ground off => free space, ground on =>
    # PEC plane at z=0.
    ground = _ground_for_engine(req, 0.0) or "free"
    return PyNECEngine(builder, ground=ground)


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


def _pynec_feed_indices(builder, currents) -> tuple[int, int]:
    """PyNECEngine returns one WireCurrents per build_wires() tuple in
    the same order, so the feed wire index is just the position of
    the first excitation-bearing tuple. Place the marker on that
    wire's centre knot — close enough to NEC's per-segment feed for a
    UI dot.
    """
    # build_wires() may emit 5-tuples with a trailing `name` field for
    # network-spec designs — permissive unpacking.
    for i, t in enumerate(builder.build_wires()):
        ev = t[3]
        name = t[4] if len(t) >= 5 else None
        # Network-spec designs source excitation off build_network() rather
        # than the `ev` field; fall back to the first named edge as the
        # visual feed location.
        if ev is not None or name is not None:
            if i >= len(currents):
                return 0, 0
            k = currents[i].knot_positions.shape[0]
            return i, k // 2
    return 0, 0


# ---------------------------------------------------------------------------
# Example factory
# ---------------------------------------------------------------------------


def _discover_variants(cls) -> tuple[str, ...]:
    """Names of every class-level `<name>_params` attribute (the variant
    convention used across the design library — e.g. `default_params`,
    `opt_params`, `z50_params`, `current_physical_params`). The
    returned list is suitable for a UI selector; the bare names (no
    `_params` suffix) are what the frontend sends back in the request.

    `default` is always first if present, so the UI lists it as the
    canonical starting point regardless of class attribute order.
    """
    suffix = "_params"
    found: list[str] = []
    for attr in dir(cls):
        if not attr.endswith(suffix) or attr.startswith("_"):
            continue
        v = getattr(cls, attr, None)
        # MappingProxyType / dict only — skip e.g. a method that happens
        # to end in _params.
        if not hasattr(v, "keys"):
            continue
        name = attr[: -len(suffix)]
        if name:
            found.append(name)
    # `default` first, rest in stable (alphabetical) order.
    found.sort(key=lambda n: (n != "default", n))
    return tuple(found)


def _serialize_param_values(params: dict) -> dict:
    """JSON-encode a params dict for shipping to the frontend.

    Complex values become {"re": ..., "im": ...} (matches the same
    shape `_rehydrate_param` accepts on the way back). Bool/int/float
    pass through. Anything exotic (None, strings that aren't enum
    options, etc.) passes through too — the frontend just ignores
    keys it doesn't have sliders for.
    """
    out: dict = {}
    for k, v in params.items():
        if isinstance(v, complex):
            out[k] = {"re": float(v.real), "im": float(v.imag)}
        else:
            out[k] = v
    return out


def _variant_params(cls, variant: str | None) -> dict:
    """Return the seed params dict for the named variant. Falls back to
    `default_params` when variant is None or doesn't resolve to an
    attribute (stale frontend, unknown name)."""
    if variant:
        params = getattr(cls, f"{variant}_params", None)
        if params is not None and hasattr(params, "keys"):
            return dict(params)
    return dict(cls.default_params)


def _ui_scalar(default_params: dict, key: str, default):
    ui = default_params.get("ui_params") or {}
    if key in ui and not isinstance(ui[key], dict):
        return ui[key]
    return default


_ARRAY_BASES = (
    Array1x2Builder,
    Array2x2Builder,
    Array1x4Builder,
    Array1x4GroupedBuilder,
    Array2x4Builder,
)


def _auto_target_z0(cls) -> float:
    """Default reference impedance for the SWR readout.

    Array designs scale 50 Ω by the element count (1×2 → 100, 2×2 → 200,
    2×4 → 400, ...) — the convention that each branch in the splitter
    sees 50 Ω after the chain of impedance transformers, so the
    combined driving point lands at N × 50.

    Everything else defaults to 50 Ω. Designs that violate either
    convention (turnstiles with per-port 50 Ω matching, designs tuned
    to 75 Ω, etc.) override via `ui_params["target_z0"]`.
    """
    if not issubclass(cls, _ARRAY_BASES):
        return 50.0
    try:
        b = cls()
        n_feeds = sum(1 for *_, ev in b.build_wires() if ev is not None)
    except Exception:
        return 50.0
    return 50.0 * max(1, n_feeds)


def _auto_multi_feed(cls) -> bool:
    """Detect whether the design has more than one excited wire.

    Builders that drive >1 feed wire in `build_wires()` get multi_feed=True
    by default — the response shape switches to include a `feeds` array
    (per-port Z + V) and the frontend renders the per-feed table.

    Designs can still force the flag via `ui_params["multi_feed"]` — set
    False to suppress the per-feed table even when multiple excitations
    exist (e.g. mirror-symmetric arrays where the per-port Z is identical
    by construction and the extra column adds no information).
    """
    try:
        b = cls()
        n_feeds = sum(1 for *_, ev in b.build_wires() if ev is not None)
    except Exception:
        return False
    return n_feeds > 1


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
    target_z0 = float(_ui_scalar(dp, "target_z0", _auto_target_z0(cls)))
    multi_feed = bool(_ui_scalar(dp, "multi_feed", _auto_multi_feed(cls)))
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
    elif isinstance(sweep_pol_raw, dict):
        # Dict form lets ui_params opt into named fields (band_locked,
        # ...) without having to supply every positional. Anchor +
        # factors fall back to the dataclass defaults when absent.
        defaults = DEFAULT_SWEEP_POLICY
        sweep_policy = SweepPolicy(
            anchor=str(sweep_pol_raw.get("anchor", defaults.anchor)),
            lo_factor=float(sweep_pol_raw.get("lo_factor", defaults.lo_factor)),
            hi_factor=float(sweep_pol_raw.get("hi_factor", defaults.hi_factor)),
            band_locked=bool(sweep_pol_raw.get("band_locked", defaults.band_locked)),
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
    has_design_freq = "design_freq" in dp
    variants = _discover_variants(cls)

    def _design_freq_default(req: dict) -> float:
        # The active variant's `freq` is the right fallback when the
        # request hasn't supplied design_freq_mhz yet — different
        # variants of one design can target different bands (e.g.
        # hexbeam's opt vs default).
        vp = _variant_params(cls, req.get("variant"))
        return float(vp.get("freq", 14.0))

    def pysim_solve(req: dict) -> dict:
        design_freq = float(req.get("design_freq_mhz", _design_freq_default(req)))
        meas_freq = float(req.get("measurement_freq_mhz", design_freq))
        builder = _build_builder(cls, req)
        builder.freq = meas_freq
        # For freq_based designs the geometry computes from
        # design_freq via build_wires(); apply the request's
        # design_freq_mhz so dragging the design-freq slider actually
        # retunes the antenna. Top-level designs don't carry the
        # parameter so the attribute write would be silently absorbed
        # into _params and never read — guard on has_design_freq.
        if has_design_freq:
            builder.design_freq = design_freq
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
            "z0_ohms": target_z0,
        }
        if multi_feed and len(zs) > 1:
            # Pull per-feed drive voltages off the engine so the frontend
            # can render each feed's phase indicator. PysimEngine stores
            # _feeds = [(polyline_idx, arclength, voltage)]; fall back to
            # 1+0j (the canonical unit drive) when missing.
            voltages = [f[2] for f in (getattr(eng, "_feeds", None) or [])]
            voltages += [complex(1.0, 0.0)] * (len(zs) - len(voltages))
            out["feeds"] = [
                {
                    "z_re": float(z.real),
                    "z_im": float(z.imag),
                    "v_re": float(v.real),
                    "v_im": float(v.imag),
                }
                for z, v in zip(zs, voltages)
            ]
        return out

    def pynec_build(req: dict) -> dict:
        # web.pynec_backend.pattern() expects this to return a build
        # dict with at least:
        #   context      — a nec_context with geometry built, ground
        #                  card applied, and excitation cards in place
        #   feed_seg     — 1-indexed segment number of the source
        #                  (only consulted by the default _run_solve()
        #                  excite path; ours supplies pynec_pattern_excite
        #                  so it's only present for parity)
        #   feed_tag     — NEC wire tag carrying the feed
        #   n_per_wire   — historical, _run_solve threads it through
        #                  but doesn't actually use it
        #   ground       — bool (informational; gn_card already on the
        #                  context)
        #   ground_fast  — bool (same)
        #   z_offset     — antenna height above ground, surfaced in
        #                  the pattern response
        #   _engine      — keep the PyNECEngine alive so the
        #                  underlying nec_context isn't released
        #                  before rp_card runs
        design_freq = float(req.get("design_freq_mhz", _design_freq_default(req)))
        meas_freq = float(req.get("measurement_freq_mhz", design_freq))
        builder = _build_builder(cls, req)
        builder.freq = meas_freq
        if has_design_freq:
            builder.design_freq = design_freq
        eng = _make_pynec_engine(req, builder)
        # Find the first excited wire to fill the feed_seg / feed_tag
        # parity fields. PyNECEngine.excitation_pairs is (tag, sub_seg,
        # voltage); take the first.
        feed_tag, feed_seg, _v = (eng.excitation_pairs or [(1, 1, 0)])[0]
        return {
            "context": eng.c,
            "feed_seg": int(feed_seg),
            "feed_tag": int(feed_tag),
            "n_per_wire": 1,
            "ground": bool(req.get("ground", False)),
            "ground_fast": False,
            "z_offset": 0.0,
            "_engine": eng,
        }

    def pynec_pattern_excite(b: dict, freq_mhz: float) -> None:
        # PyNECEngine already applied the gn_card and ex_card during
        # _build_geometry, so the pattern endpoint only needs to set
        # the frequency and execute. Reusing _run_solve() would add a
        # second ex_card on top of the one already in place.
        c = b["context"]
        c.fr_card(0, 1, float(freq_mhz), 0)
        c.xq_card(0)

    def pynec_solve(req: dict) -> dict:
        # Mirror pysim_solve but route through PyNECEngine. Response
        # shape is identical so the frontend renders the result the
        # same way; the `solver` field gets stamped to "pynec" by
        # server.solve()'s outer wrapper.
        design_freq = float(req.get("design_freq_mhz", _design_freq_default(req)))
        meas_freq = float(req.get("measurement_freq_mhz", design_freq))
        builder = _build_builder(cls, req)
        builder.freq = meas_freq
        if has_design_freq:
            builder.design_freq = design_freq
        eng = _make_pynec_engine(req, builder)
        t0 = time.perf_counter()
        zs = eng.impedance()
        currents = eng.current_distribution()
        solve_ms = (time.perf_counter() - t0) * 1e3
        feed_wire_idx, feed_knot_idx = _pynec_feed_indices(builder, currents)
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
            "z0_ohms": target_z0,
        }
        if multi_feed and len(zs) > 1:
            # PyNECEngine.excitation_pairs is [(tag, sub_seg, voltage)];
            # pull the voltage off each so per-feed phase comes through.
            voltages = [v for _t, _s, v in (eng.excitation_pairs or [])]
            voltages += [complex(1.0, 0.0)] * (len(zs) - len(voltages))
            out["feeds"] = [
                {
                    "z_re": float(z.real),
                    "z_im": float(z.imag),
                    "v_re": float(v.real),
                    "v_im": float(v.imag),
                }
                for z, v in zip(zs, voltages)
            ]
        return out

    def pysim_sweep(req: dict, freqs_mhz: list[float]):
        builder = _build_builder(cls, req)
        # PysimEngine reads builder.freq only for the initial wavelength
        # passed to _make_solver — impedance_sweep overrides k per point.
        builder.freq = float(freqs_mhz[0]) if freqs_mhz else float(builder.freq)
        # Geometry is fixed across the sweep; honour the request's
        # design_freq so the sweep sees the same antenna the live
        # solve sees. See pysim_solve for the rationale.
        if has_design_freq:
            builder.design_freq = float(
                req.get("design_freq_mhz", _design_freq_default(req))
            )
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
        pynec_solve=pynec_solve,
        pynec_build=pynec_build,
        pynec_pattern_excite=pynec_pattern_excite,
        multi_feed=multi_feed,
        param_schema=param_schema,
        bands=bands,
        meas_freq_range_mhz=tuple(meas_range) if meas_range else None,
        sweep_policy=sweep_policy,
        default_view=default_view,
        default_freq_mhz=float(dp["freq"]) if "freq" in dp else None,
        has_design_freq=has_design_freq,
        variants=variants,
        variant_values={
            v: _serialize_param_values(_strip_ui(_variant_params(cls, v)))
            for v in variants
        },
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
