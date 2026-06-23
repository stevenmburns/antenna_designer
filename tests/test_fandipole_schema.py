"""Schema-level checks for fandipole's ui_params + variants.

Covers the wiring between Builder.default_params['ui_params'] and the
web adapter's derived ParamSpec/ParamGroupSpec/SweepPolicy: the bands
tuple-of-dicts must surface as a repeating group whose freq leaf drives
measFreq, the sweep policy must arrive anchored on measFreq with
band_locked=True, and the three intended variants must be discovered
with the expected n_bands counts.
"""

import web.examples  # noqa: F401  — registers examples; primes the adapter
from antennaknobs.designs.multiband.fandipole import Builder
from web.adapter import _derive_schema, _make_example
from web.examples._base import ParamGroupSpec, ParamSpec


def _schema_by_name():
    schema = _derive_schema(Builder.default_params)
    return {s.name: s for s in schema}


def test_bands_emerges_as_group():
    by_name = _schema_by_name()
    bands = by_name["bands"]
    assert isinstance(bands, ParamGroupSpec)
    assert bands.repeat_count == "n_bands"
    assert bands.max_repeats == 5
    assert bands.link_meas_freq_to_param == "freq"
    assert "{i}" in bands.label_template


def test_bands_inner_params_are_freq_and_length_factor():
    bands = _schema_by_name()["bands"]
    leaf_names = {p.name for p in bands.params}
    assert leaf_names == {"freq", "length_factor"}
    for p in bands.params:
        assert isinstance(p, ParamSpec)


def test_bands_default_overrides_seed_five_bands_in_canonical_order():
    bands = _schema_by_name()["bands"]
    assert len(bands.default_overrides) == 5
    freqs = [d["freq"] for d in bands.default_overrides]
    # 20m → 10m, canonical ordering.
    assert freqs == [14.300, 18.1575, 21.383, 24.97, 28.47]


def test_n_bands_scalar_present_for_repeat_count():
    by_name = _schema_by_name()
    n_bands = by_name["n_bands"]
    assert isinstance(n_bands, ParamSpec)
    assert n_bands.kind == "int"
    assert n_bands.max == 5
    assert n_bands.min == 1


def test_sweep_policy_is_band_locked_on_meas_freq():
    ex = _make_example("fandipole", Builder)
    assert ex.sweep_policy.anchor == "meas_freq"
    assert ex.sweep_policy.band_locked is True


def test_variants_cover_five_band_and_pair_variants():
    ex = _make_example("fandipole", Builder)
    assert set(ex.variants) >= {"default", "five_band", "pair_17_15", "pair_12_10"}
    assert ex.variant_values["pair_17_15"]["n_bands"] == 2
    assert ex.variant_values["pair_12_10"]["n_bands"] == 2
    assert ex.variant_values["five_band"]["n_bands"] == 5


def test_pair_variants_keep_full_bands_tuple_for_clean_overlay():
    # Variants always carry a length-5 bands array so the frontend's
    # wholesale-overlay variant switch stays aligned with the schema's
    # max_repeats-preallocated instances; n_bands selects how many
    # render but the others are available if the user bumps the count.
    ex = _make_example("fandipole", Builder)
    for v in ("pair_17_15", "pair_12_10", "five_band"):
        assert len(ex.variant_values[v]["bands"]) == 5


def test_pair_17_15_first_two_active_bands_are_17m_and_15m():
    ex = _make_example("fandipole", Builder)
    bands = ex.variant_values["pair_17_15"]["bands"]
    assert bands[0]["freq"] == 18.1575
    assert bands[1]["freq"] == 21.383


def test_pair_12_10_first_two_active_bands_are_12m_and_10m():
    ex = _make_example("fandipole", Builder)
    bands = ex.variant_values["pair_12_10"]["bands"]
    assert bands[0]["freq"] == 24.97
    assert bands[1]["freq"] == 28.47


def test_bands_cover_each_seeded_freq():
    # band_locked sweep only kicks in when measFreq sits inside a
    # configured band. Guard against a future refactor that drops the
    # default HF bands from the fandipole example.
    ex = _make_example("fandipole", Builder)
    bands = _schema_by_name()["bands"]
    for d in bands.default_overrides:
        f = float(d["freq"])
        assert any(b.min_mhz <= f <= b.max_mhz for b in ex.bands), (
            f"freq {f} falls outside every band in ex.bands"
        )
