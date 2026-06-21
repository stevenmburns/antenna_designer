import pytest

from antenna_designer.cli import get_builder, list_variants
from antenna_designer.designs.beams import hexbeam, moxon
from antenna_designer.designs.multiband import twoband_fan_dipole
from antenna_designer.designs.loops import delta_loop
from antenna_designer.designs.specialty import hentenna


def _design_params(inst):
    """Builder _params with the framework keys (nominal_nsegs, ...)
    stripped. Variant resolution is about design params only."""
    return {k: v for k, v in inst._params.items() if k not in inst.FRAMEWORK_PARAMS}


def test_no_colon_uses_default_params():
    factory = get_builder("beams.hexbeam")
    inst = factory()
    assert _design_params(inst) == dict(hexbeam.Builder.default_params)


def test_explicit_default_variant():
    factory = get_builder("hexbeam:default")
    inst = factory()
    assert _design_params(inst) == dict(hexbeam.Builder.default_params)


def test_named_variant_resolves():
    factory = get_builder("hexbeam:opt")
    inst = factory()
    assert _design_params(inst) == dict(hexbeam.Builder.opt_params)


def test_variant_on_moxon_original():
    factory = get_builder("moxon:original")
    inst = factory()
    assert _design_params(inst) == dict(moxon.Builder.original_params)


def test_renamed_twoband_variant():
    factory = get_builder("twoband_fan_dipole:s07")
    inst = factory()
    assert _design_params(inst) == dict(twoband_fan_dipole.Builder.s07_params)


def test_renamed_specialty_variant():
    factory = get_builder("specialty.hentenna:z100")
    inst = factory()
    assert _design_params(inst) == dict(hentenna.Builder.z100_params)


def test_renamed_loop_variant():
    factory = get_builder("loops.delta_loop:z200")
    inst = factory()
    assert _design_params(inst) == dict(delta_loop.Builder.z200_params)


def test_unknown_variant_raises_with_available():
    with pytest.raises(ValueError) as exc:
        get_builder("hexbeam:does_not_exist")
    msg = str(exc.value)
    assert "does_not_exist" in msg
    assert "opt" in msg
    assert "default" in msg


def test_list_variants_for_hexbeam():
    assert list_variants(hexbeam.Builder) == ["default", "opt"]


def test_list_variants_for_moxon():
    assert list_variants(moxon.Builder) == ["default", "opt", "original"]
