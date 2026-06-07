"""Builder discovery + parameter-schema generation for the web UI.

Walks the designs/ subpackage and emits a JSON-friendly description of
every concrete Builder: its dotted name, its variants, and its slider
schema (per-parameter default + suggested range).

Range inference is heuristic by parameter name, with a generic ±50%
fallback. Designs can override per-parameter by declaring an optional
class attribute `param_ranges = {"key": (lo, hi), ...}` or
`param_ranges = {"key": (lo, hi, step), ...}`.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from importlib import import_module
from typing import Any

from . import __file__ as _pkg_init
from .cli import list_variants

_DESIGNS_ROOT = pathlib.Path(_pkg_init).parent / "designs"

# Parameter-name-based range heuristics. Keys match by exact name first,
# then by suffix (so `slope_top`, `slope_bot` etc. all map to `slope`).
# Values are (min, max, step).
_RANGE_BY_NAME: dict[str, tuple[float, float, float]] = {
    "freq": (1.0, 60.0, 0.01),
    "design_freq": (1.0, 60.0, 0.01),
    "measurement_freq": (1.0, 60.0, 0.01),
    "base": (0.5, 20.0, 0.05),
    "length": (0.5, 30.0, 0.01),
    "length_factor": (0.3, 1.5, 0.001),
    "halfdriver_factor": (0.2, 0.6, 0.001),
    "slope": (0.0, 1.5, 0.001),
    "aspect_ratio": (0.1, 2.0, 0.001),
    "angle_radians": (0.0, 1.5708, 0.001),
    "angle_deg": (0.0, 90.0, 0.1),
    "phase_lr": (0.0, 360.0, 1.0),
    "phase_tb": (0.0, 360.0, 1.0),
    "del_y": (0.5, 15.0, 0.05),
    "del_z": (0.0, 10.0, 0.05),
    "del_y0": (0.5, 15.0, 0.05),
    "del_y1": (0.5, 15.0, 0.05),
    "n_directors": (0, 30, 1),
    "spacing_wavelengths": (0.05, 0.5, 0.001),
}


def _range_for(name: str, default: Any) -> tuple[float, float, float]:
    """Pick (min, max, step) for a parameter by name, falling back to
    ±50% around the default value."""
    if name in _RANGE_BY_NAME:
        return _RANGE_BY_NAME[name]
    # Suffix fallback — slope_top → slope, length_top → length, etc.
    for suffix in ("_top", "_bot", "_itop", "_ibot", "_otop", "_obot"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            if stem in _RANGE_BY_NAME:
                return _RANGE_BY_NAME[stem]
    # Generic fallback.
    try:
        d = float(default)
    except (TypeError, ValueError):
        return (0.0, 1.0, 0.01)
    if d == 0:
        return (-1.0, 1.0, 0.01)
    half = abs(d) * 0.5
    lo, hi = d - half, d + half
    if isinstance(default, int) and not isinstance(default, bool):
        return (int(lo), int(hi), 1)
    step = max(abs(d) / 200.0, 1e-4)
    return (float(lo), float(hi), float(step))


def _param_schema(cls: type, params: Mapping[str, Any]) -> list[dict]:
    """Build the slider schema for one (class, variant-params) pair.

    Honours an optional `param_ranges` class attribute that overrides
    the heuristic for specific keys: {"key": (lo, hi)} or
    {"key": (lo, hi, step)}.
    """
    overrides: Mapping[str, tuple] = getattr(cls, "param_ranges", {}) or {}
    out = []
    for key, default in params.items():
        if isinstance(default, complex):
            # Complex-valued params (e.g. excitation voltage) get two-input
            # re/im treatment in the UI rather than a single slider.
            out.append(
                {
                    "key": key,
                    "default": {"re": float(default.real), "im": float(default.imag)},
                    "is_complex": True,
                }
            )
            continue
        if key in overrides:
            ov = overrides[key]
            lo, hi = float(ov[0]), float(ov[1])
            step = float(ov[2]) if len(ov) >= 3 else (hi - lo) / 200.0
        else:
            lo, hi, step = _range_for(key, default)
        out.append(
            {
                "key": key,
                "default": float(default)
                if isinstance(default, (int, float))
                else default,
                "min": float(lo),
                "max": float(hi),
                "step": float(step),
                "is_int": isinstance(default, int) and not isinstance(default, bool),
            }
        )
    return out


def _iter_design_modules() -> list[str]:
    """Yield dotted module names (relative to antenna_designer.designs)
    for every .py file under designs/, skipping package __init__s."""
    names = []
    for path in sorted(_DESIGNS_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(_DESIGNS_ROOT).with_suffix("")
        names.append(".".join(rel.parts))
    return names


def _try_concrete(cls: type) -> bool:
    """A Builder is 'concrete' when zero-arg construction succeeds.
    Excludes abstract array bases like Array1x2Builder that need an
    element_builder positional arg."""
    if not hasattr(cls, "default_params"):
        return False
    try:
        cls()
    except TypeError:
        return False
    except Exception:
        return False
    return True


def list_builders() -> list[dict]:
    """Enumerate every concrete Builder under antenna_designer.designs.

    Returns a list of dicts:
        {
          "name": "freq_based.dipole_turnstile",
          "variants": ["default", ...],
          "params": [{key, default, min, max, step, is_int}, ...],
        }
    The 'params' field reflects the default-variant param set; variant
    schemas can be fetched separately via builder_schema(name, variant).
    """
    out = []
    for dotted in _iter_design_modules():
        try:
            mod = import_module(f"antenna_designer.designs.{dotted}")
        except Exception:
            continue
        cls = getattr(mod, "Builder", None)
        if cls is None or not _try_concrete(cls):
            continue
        variants = ["default", *[v for v in list_variants(cls) if v != "default"]]
        try:
            b = cls()
        except Exception:
            continue
        out.append(
            {
                "name": dotted,
                "variants": variants,
                "params": _param_schema(cls, b._params),
            }
        )
    return out


def builder_schema(name: str, variant: str = "default") -> dict | None:
    """Return one builder's schema, resolving the named variant."""
    try:
        mod = import_module(f"antenna_designer.designs.{name}")
    except ModuleNotFoundError:
        return None
    cls = getattr(mod, "Builder", None)
    if cls is None or not _try_concrete(cls):
        return None
    if variant and variant != "default":
        attr = f"{variant}_params"
        params_src = getattr(cls, attr, None)
        if params_src is None:
            return None
        b = cls(params=dict(params_src))
    else:
        b = cls()
    return {
        "name": name,
        "variant": variant,
        "variants": ["default", *[v for v in list_variants(cls) if v != "default"]],
        "params": _param_schema(cls, b._params),
    }
