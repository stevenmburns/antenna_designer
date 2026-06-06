import pytest

from antenna_designer.cli import get_builder, list_variants
from antenna_designer.designs import hexbeam, moxon, twoband_fan_dipole
from antenna_designer.designs.freq_based import hentenna, delta_loop


def test_no_colon_uses_default_params():
    factory = get_builder('hexbeam')
    inst = factory()
    assert dict(inst._params) == dict(hexbeam.Builder.default_params)


def test_explicit_default_variant():
    factory = get_builder('hexbeam:default')
    inst = factory()
    assert dict(inst._params) == dict(hexbeam.Builder.default_params)


def test_named_variant_resolves():
    factory = get_builder('hexbeam:opt')
    inst = factory()
    assert dict(inst._params) == dict(hexbeam.Builder.opt_params)


def test_variant_on_moxon_original():
    factory = get_builder('moxon:original')
    inst = factory()
    assert dict(inst._params) == dict(moxon.Builder.original_params)


def test_renamed_twoband_variant():
    factory = get_builder('twoband_fan_dipole:s07')
    inst = factory()
    assert dict(inst._params) == dict(twoband_fan_dipole.Builder.s07_params)


def test_renamed_freq_based_variant():
    factory = get_builder('freq_based.hentenna:z100')
    inst = factory()
    assert dict(inst._params) == dict(hentenna.Builder.z100_params)


def test_renamed_loop_variant():
    factory = get_builder('freq_based.delta_loop:z200')
    inst = factory()
    assert dict(inst._params) == dict(delta_loop.Builder.z200_params)


def test_unknown_variant_raises_with_available():
    with pytest.raises(ValueError) as exc:
        get_builder('hexbeam:does_not_exist')
    msg = str(exc.value)
    assert 'does_not_exist' in msg
    assert 'opt' in msg
    assert 'default' in msg


def test_list_variants_for_hexbeam():
    assert list_variants(hexbeam.Builder) == ['default', 'opt']


def test_list_variants_for_moxon():
    assert list_variants(moxon.Builder) == ['default', 'opt', 'original']
