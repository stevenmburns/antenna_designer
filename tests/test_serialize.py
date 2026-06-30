"""Tests for antennaknobs.serialize — knob values → paste-ready Python source.

Every emitted block must be valid Python that, when executed, reproduces the
input values (up to the requested display precision). The round-trip is checked
by ``exec``-ing the source and comparing the resulting binding.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from antennaknobs.serialize import (
    builder_params_source,
    params_source,
)


def _eval_block(src: str, name: str = "default_params"):
    """Exec an emitted assignment block and return the bound value."""
    ns: dict = {"MappingProxyType": MappingProxyType}
    exec(src, ns)
    return ns[name]


def test_simple_dict_round_trips():
    params = {"freq": 14.0, "base": 10.0, "length_factor": 0.97}
    src = params_source(params)
    assert src.startswith("default_params = {")
    assert _eval_block(src) == params


def test_emitted_source_is_valid_python_and_named():
    src = params_source({"freq": 28.57}, name="opt_params")
    assert src.startswith("opt_params = {")
    assert _eval_block(src, "opt_params") == {"freq": 28.57}


def test_floats_default_to_shortest_round_trip():
    # No precision metadata: the exact value must survive, not be rounded away.
    params = {"halfdriver": 2.4597430629596713}
    out = _eval_block(params_source(params))
    assert out["halfdriver"] == 2.4597430629596713


def test_per_knob_precision_rounds_for_display():
    params = {"z0_match": 450.123456, "match_len_frac": 0.461234}
    src = params_source(params, precision={"z0_match": 1, "match_len_frac": 3})
    assert "'z0_match': 450.1," in src
    assert "'match_len_frac': 0.461," in src


def test_default_precision_applies_when_no_per_knob_entry():
    src = params_source({"x": 0.123456789}, default_precision=4)
    assert "'x': 0.1235," in src


def test_trailing_zeros_trimmed_but_keeps_one_decimal():
    src = params_source({"a": 14.0, "b": 1.2000}, precision={"a": 4, "b": 4})
    assert "'a': 14.0," in src
    assert "'b': 1.2," in src


def test_ints_bools_strings_render_natively():
    params = {"n": 5, "daisy_chain": True, "view": "xz"}
    src = params_source(params)
    assert "'n': 5," in src
    assert "'daisy_chain': True," in src  # not 1
    assert "'view': 'xz'," in src
    assert _eval_block(src) == params


def test_complex_excitation_round_trips():
    params = {"drive": 1 + 0j}
    out = _eval_block(params_source(params))
    assert out["drive"] == 1 + 0j


def test_nested_ui_params_round_trips():
    params = {
        "freq": 14.0,
        "ui_params": {
            "default_view": "xz",
            "z0_match": {"min": 200.0, "max": 600.0},
        },
    }
    out = _eval_block(params_source(params))
    assert out == params


def test_include_ui_false_drops_ui_params():
    params = {"freq": 14.0, "ui_params": {"default_view": "xz"}}
    src = params_source(params, include_ui=False)
    assert "ui_params" not in src
    assert _eval_block(src) == {"freq": 14.0}


def test_bands_tuple_of_dicts_round_trips():
    params = {
        "n_bands": 2,
        "bands": (
            {"freq": 14.3, "halfdriver_factor": 1.071},
            {"freq": 21.383, "halfdriver_factor": 1.071},
        ),
    }
    out = _eval_block(params_source(params))
    assert out == params
    # tuple stays a tuple, not silently widened to a list
    assert isinstance(out["bands"], tuple)


def test_mappingproxy_wrap():
    src = params_source({"freq": 14.0}, wrap="mappingproxy")
    assert src.startswith("default_params = MappingProxyType({")
    assert _eval_block(src) == MappingProxyType({"freq": 14.0})


def test_unknown_wrap_raises():
    with pytest.raises(ValueError, match="unknown wrap"):
        params_source({"freq": 14.0}, wrap="tuple")


# --- builder_params_source ----------------------------------------------------


def test_builder_params_source_drops_framework_params():
    import antennaknobs as ant

    class Builder(ant.AntennaBuilder):
        default_params = MappingProxyType({"freq": 14.0, "base": 10.0})

        def build_wires(self):
            return []

    b = Builder()
    # nominal_nsegs is a framework param merged into _params; it must not leak.
    assert "nominal_nsegs" in b._params
    src = builder_params_source(b)
    assert "nominal_nsegs" not in src
    assert _eval_block(src) == {"freq": 14.0, "base": 10.0}


def test_builder_params_source_applies_ui_precision():
    import antennaknobs as ant

    class Builder(ant.AntennaBuilder):
        default_params = MappingProxyType(
            {
                "z0_match": 450.0,
                "ui_params": MappingProxyType(
                    {"z0_match": {"step": 1.0, "precision": 1}}
                ),
            }
        )

        def build_wires(self):
            return []

    b = Builder()
    b.z0_match = 451.98765
    src = builder_params_source(b)
    assert "'z0_match': 452.0," in src


def test_builder_params_source_reflects_runtime_edits():
    import antennaknobs as ant

    class Builder(ant.AntennaBuilder):
        default_params = MappingProxyType({"length_factor": 1.0})

        def build_wires(self):
            return []

    b = Builder()
    b.length_factor = 0.965
    out = _eval_block(builder_params_source(b))
    assert out["length_factor"] == 0.965
