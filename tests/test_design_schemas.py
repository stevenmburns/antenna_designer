"""Per-antenna schema unit tests.

For every Builder registered by web.adapter, assert the derived
ParamSpec / variants / sweep policy / view shape the frontend depends on.
This is the generic cousin of test_fandipole_schema.py — fandipole keeps
its bespoke tests for the bands-group wiring; everything else is covered
here with one parametrized sweep.

Targeted tests at the bottom pin down design-specific UI choices
(yagi.n_directors as an int slider, hentenna_slant's tight precision
overrides, hexbeam/moxon variant lists, twoband_fan_dipole's sweep
variant family).
"""

from __future__ import annotations

import importlib

import pytest

import web.examples  # noqa: F401 — primes the adapter
from web.adapter import _derive_schema, _make_example
from web.examples import REGISTRY
from web.examples._base import (
    DEFAULT_HF_BANDS,
    AntennaExample,
    BandSpec,
    ParamGroupSpec,
    ParamSpec,
    SweepPolicy,
)


# ---------------------------------------------------------------------------
# Generic coverage — every registered design
# ---------------------------------------------------------------------------


DESIGN_NAMES = sorted(REGISTRY.keys())


def _builder_cls(name: str):
    mod = importlib.import_module(f"antenna_designer.designs.{name}")
    return mod.Builder


@pytest.mark.parametrize("name", DESIGN_NAMES)
def test_schema_excludes_freq_and_design_freq_sliders(name):
    ex = REGISTRY[name]
    slider_names = {s.name for s in ex.param_schema}
    # The dedicated meas-freq slider and the design-freq band-tab row own
    # these — the schema must never duplicate them.
    assert "freq" not in slider_names
    assert "design_freq" not in slider_names


@pytest.mark.parametrize("name", DESIGN_NAMES)
def test_schema_covers_every_non_freq_default_param(name):
    cls = _builder_cls(name)
    dp = dict(cls.default_params)
    slider_names = {s.name for s in REGISTRY[name].param_schema}
    for key, val in dp.items():
        if key in ("ui_params", "freq", "design_freq"):
            continue
        if isinstance(val, complex):
            # Complex defaults intentionally have no UI; settable via
            # request body's {re, im} shape only.
            continue
        if isinstance(val, str):
            # String defaults need an enum_options override to surface.
            continue
        assert key in slider_names, f"{name}: missing slider for {key!r}"


@pytest.mark.parametrize("name", DESIGN_NAMES)
def test_default_variant_listed_first(name):
    variants = REGISTRY[name].variants
    assert variants, f"{name}: no variants discovered"
    assert variants[0] == "default"


@pytest.mark.parametrize("name", DESIGN_NAMES)
def test_variant_values_serialise_for_every_variant(name):
    ex = REGISTRY[name]
    for v in ex.variants:
        assert v in ex.variant_values, f"{name}: missing values for variant {v!r}"
        # ui_params is solver-internal; the wire to the frontend must
        # not carry it (variant_values goes straight to App.tsx).
        assert "ui_params" not in ex.variant_values[v]


@pytest.mark.parametrize("name", DESIGN_NAMES)
def test_default_view_is_a_valid_2d_plane(name):
    assert REGISTRY[name].default_view in {"xy", "yz", "xz"}


@pytest.mark.parametrize("name", DESIGN_NAMES)
def test_sweep_policy_is_well_formed(name):
    sp = REGISTRY[name].sweep_policy
    assert isinstance(sp, SweepPolicy)
    assert sp.anchor in {"design_freq", "meas_freq"}
    assert sp.lo_factor > 0 and sp.hi_factor > sp.lo_factor


@pytest.mark.parametrize("name", DESIGN_NAMES)
def test_bands_default_to_hf_set_unless_overridden(name):
    bands = REGISTRY[name].bands
    assert all(isinstance(b, BandSpec) for b in bands)
    # No design currently zeroes out the band row — guard against an
    # accidental ui_params['bands'] = () regressing the design-freq UI.
    assert len(bands) >= 1
    # Defaulted designs share the HF set object; overrides have their
    # own tuple but still parse as BandSpecs (covered above).
    if "ui_params" not in dict(_builder_cls(name).default_params):
        assert bands is DEFAULT_HF_BANDS


@pytest.mark.parametrize("name", DESIGN_NAMES)
def test_param_schema_specs_are_typed_correctly(name):
    for spec in REGISTRY[name].param_schema:
        assert isinstance(spec, (ParamSpec, ParamGroupSpec))
        if isinstance(spec, ParamSpec) and spec.kind in ("float", "int"):
            assert spec.min is not None and spec.max is not None
            assert spec.min < spec.max


@pytest.mark.parametrize("name", DESIGN_NAMES)
def test_example_round_trips_through_make_example(name):
    # Re-deriving the example mid-test shakes out hidden mutation in
    # _make_example (closures over Builder dicts that get mutated on
    # subsequent solves, etc.).
    cls = _builder_cls(name)
    ex = _make_example(name, cls)
    assert isinstance(ex, AntennaExample)
    assert ex.name == name


# ---------------------------------------------------------------------------
# freq_based.yagi — n_directors is the integer scalar that drives the
# director count in build_wires().
# ---------------------------------------------------------------------------


def test_yagi_n_directors_is_int_slider():
    schema = {s.name: s for s in REGISTRY["freq_based.yagi"].param_schema}
    n_dir = schema["n_directors"]
    assert isinstance(n_dir, ParamSpec)
    assert n_dir.kind == "int"
    assert n_dir.default == 2


def test_yagi_factor_sliders_present():
    schema = {s.name: s for s in REGISTRY["freq_based.yagi"].param_schema}
    for key in ("length_factor", "director_factor", "reflector_factor", "boom_factor"):
        assert key in schema
        assert schema[key].kind == "float"


# ---------------------------------------------------------------------------
# freq_based.hentenna_slant — explicit ui_params overrides on
# length_factor, top_aspect, bot_aspect, slant_degrees.
# ---------------------------------------------------------------------------


def test_hentenna_slant_aspect_overrides_applied():
    schema = {s.name: s for s in REGISTRY["freq_based.hentenna_slant"].param_schema}
    top = schema["top_aspect"]
    bot = schema["bot_aspect"]
    slant = schema["slant_degrees"]
    assert (top.min, top.max) == (0.5, 4.5)
    assert (bot.min, bot.max) == (0.0, 2.0)
    assert (slant.min, slant.max) == (0.0, 45.0)
    assert top.precision == 4 and bot.precision == 4
    assert slant.precision == 0


def test_hentenna_slant_lists_z50_z100_variants():
    variants = REGISTRY["freq_based.hentenna_slant"].variants
    assert set(variants) >= {"default", "z50", "z100"}


def test_hentenna_slant_z100_overrides_default_factors():
    vv = REGISTRY["freq_based.hentenna_slant"].variant_values
    assert vv["z100"]["length_factor"] != vv["default"]["length_factor"]
    assert vv["z100"]["top_aspect"] != vv["default"]["top_aspect"]


# ---------------------------------------------------------------------------
# Variant family pinning — designs whose variants are the user-facing
# selectors. Catches an accidental rename of a class attribute.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("hexbeam", {"default", "opt"}),
        ("moxon", {"default", "opt", "original"}),
        ("invveearray", {"default", "old"}),
        ("freq_based.delta_loop", {"default", "z100", "z200"}),
        ("freq_based.diamond_loop", {"default", "z100", "z200"}),
        ("freq_based.inv_delta_loop", {"default", "z100", "z200"}),
        ("freq_based.hentenna", {"default", "z50", "z100"}),
        ("freq_based.delta_loop_slanted", {"default", "slant0", "slant30"}),
        ("delta_looparray", {"default", "dy3_dz2", "dy35_dz2", "dy45_dz2"}),
    ],
)
def test_variant_family(name, expected):
    assert set(REGISTRY[name].variants) >= expected


def test_twoband_fan_dipole_carries_spacing_sweep_variants():
    # twoband_fan_dipole sweeps spacing factor across s01..s07 + an
    # eps-perturbed variant. The full set is what feeds the UI's
    # variant selector — losing any of them silently breaks the
    # comparison sweep the user runs after picking a baseline.
    expected = {
        "default",
        "current_physical",
        "s01",
        "s015",
        "s01_eps001",
        "s02",
        "s025",
        "s03",
        "s05",
        "s07",
    }
    assert set(REGISTRY["twoband_fan_dipole"].variants) >= expected


# ---------------------------------------------------------------------------
# design_freq presence — freq_based.* designs derive geometry from
# design_freq; top-level designs are hand-tuned in absolute meters. The
# adapter uses has_design_freq to decide whether to write the request's
# design_freq_mhz onto the builder; the wrong polarity silently breaks
# every solve.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DESIGN_NAMES)
def test_has_design_freq_matches_default_params(name):
    cls = _builder_cls(name)
    expected = "design_freq" in dict(cls.default_params)
    assert REGISTRY[name].has_design_freq is expected
