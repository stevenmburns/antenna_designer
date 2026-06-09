"""Schema-level checks for fandipole's ui_params.

Covers the wiring between Builder.default_params['ui_params'] and the
web adapter's derived ParamSpec / SweepPolicy: the length_factor_NN
sliders must carry a link_meas_freq_to_param pointing at the matching
freq_NN, the freq_NN sliders must self-link, and the sweep policy must
come through anchored on measFreq with band_locked=True.
"""

import web.examples  # noqa: F401  — registers examples; primes the adapter
from antenna_designer.designs.freq_based.fandipole import Builder
from web.adapter import _derive_schema, _make_example


_BAND_PAIRS = [
    ("length_factor_20", "freq_20"),
    ("length_factor_17", "freq_17"),
    ("length_factor_15", "freq_15"),
    ("length_factor_12", "freq_12"),
    ("length_factor_10", "freq_10"),
]


def _schema_by_name():
    schema = _derive_schema(Builder.default_params)
    return {s.name: s for s in schema}


def test_length_sliders_link_to_matching_freq():
    specs = _schema_by_name()
    for length_name, freq_name in _BAND_PAIRS:
        assert specs[length_name].link_meas_freq_to_param == freq_name


def test_freq_sliders_self_link():
    specs = _schema_by_name()
    for _, freq_name in _BAND_PAIRS:
        assert specs[freq_name].link_meas_freq_to_param == freq_name


def test_unrelated_param_has_no_meas_link():
    specs = _schema_by_name()
    assert specs["slope"].link_meas_freq_to_param is None


def test_sweep_policy_is_band_locked_on_meas_freq():
    ex = _make_example("fandipole", Builder)
    assert ex.sweep_policy.anchor == "meas_freq"
    assert ex.sweep_policy.band_locked is True


def test_bands_cover_each_freq_default():
    # The band-locked sweep only kicks in when measFreq sits inside a
    # configured band. Guard against a future refactor that drops the
    # default HF bands from the fandipole example, which would silently
    # break the per-band sweep snap.
    ex = _make_example("fandipole", Builder)
    for _, freq_name in _BAND_PAIRS:
        f = float(Builder.default_params[freq_name])
        assert any(b.min_mhz <= f <= b.max_mhz for b in ex.bands), (
            f"{freq_name}={f} falls outside every band in ex.bands"
        )
