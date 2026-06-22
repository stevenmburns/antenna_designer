import argparse

import pytest

import antenna_designer as ant
from antenna_designer.cli import (
    PYSIM_BASES,
    parse_engine_spec,
    make_engine_factory,
    broadcast_pairs,
    _GROUND_UNSET,
)
from antenna_designer.engines import PyNECEngine, PysimEngine
from pysim import TriangularPySim, SinusoidalPySim, BSplinePySim

from conftest import needs_pynec


@needs_pynec
def test_parse_pynec_no_basis():
    assert parse_engine_spec("pynec") == ("pynec", {})


def test_parse_pysim_default():
    assert parse_engine_spec("pysim") == ("pysim", {})


@pytest.mark.parametrize(
    "basis,cls",
    [
        ("triangular", TriangularPySim),
        ("sinusoidal", SinusoidalPySim),
        ("bspline", BSplinePySim),
    ],
)
def test_parse_pysim_with_basis(basis, cls):
    name, kw = parse_engine_spec(f"pysim:{basis}")
    assert name == "pysim"
    assert kw == {"solver": cls}


def test_parse_unknown_engine_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_engine_spec("bogus")


def test_parse_pynec_with_basis_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_engine_spec("pynec:triangular")


def test_parse_pysim_unknown_basis_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_engine_spec("pysim:not_a_basis")


@needs_pynec
def test_make_factory_returns_class_when_no_kwargs():
    assert make_engine_factory("pynec", _GROUND_UNSET) is PyNECEngine
    assert make_engine_factory("pysim", _GROUND_UNSET) is PysimEngine


def test_make_factory_binds_solver():
    factory = make_engine_factory("pysim:sinusoidal", _GROUND_UNSET)
    assert factory.func is PysimEngine
    assert factory.keywords == {"solver": SinusoidalPySim}


def test_make_factory_binds_ground_and_solver():
    factory = make_engine_factory("pysim:bspline", "pec")
    assert factory.func is PysimEngine
    assert factory.keywords == {"solver": BSplinePySim, "ground": "pec"}


def test_pysim_bases_keys():
    assert set(PYSIM_BASES) == {
        "triangular",
        "sinusoidal",
        "bspline",
        "hmatrix",
        "arrayblock",
    }


O = " --fn /dev/null"


@needs_pynec
def test_cli_compare_patterns_multi_engine():
    ant.cli(
        f"compare_patterns --builders dipoles.invvee:dipole --engines pynec pysim{O}".split()
    )


@needs_pynec
def test_cli_compare_patterns_single_engine_still_works():
    ant.cli(
        f"compare_patterns --builders dipoles.invvee:dipole dipoles.invvee --engines pynec{O}".split()
    )


def test_cli_compare_patterns_pysim_basis():
    ant.cli(
        f"compare_patterns --builders dipoles.invvee:dipole --engines pysim:triangular pysim:sinusoidal{O}".split()
    )


def test_broadcast_equal_length():
    assert broadcast_pairs(["a", "b", "c"], ["x", "y", "z"]) == [
        ("a", "x"),
        ("b", "y"),
        ("c", "z"),
    ]


def test_broadcast_single_engine():
    assert broadcast_pairs(["a", "b", "c"], ["x"]) == [
        ("a", "x"),
        ("b", "x"),
        ("c", "x"),
    ]


def test_broadcast_single_builder():
    assert broadcast_pairs(["a"], ["x", "y", "z"]) == [
        ("a", "x"),
        ("a", "y"),
        ("a", "z"),
    ]


def test_broadcast_mismatch_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        broadcast_pairs(["a", "b"], ["x", "y", "z"])


@needs_pynec
def test_cli_compare_patterns_three_by_three_paired():
    ant.cli(
        f"compare_patterns --builders dipoles.invvee:dipole dipoles.invvee specialty.bowtie "
        f"--engines pynec pysim:triangular pysim:sinusoidal{O}".split()
    )


def test_cli_compare_patterns_three_builders_one_engine():
    ant.cli(
        f"compare_patterns --builders dipoles.invvee:dipole dipoles.invvee specialty.bowtie --engines pysim{O}".split()
    )


def test_cli_compare_patterns_mismatch_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        ant.cli(
            f"compare_patterns --builders dipoles.invvee:dipole dipoles.invvee --engines pynec pysim:triangular pysim:sinusoidal{O}".split()
        )


def test_cli_pattern_with_basis_spec():
    ant.cli(
        f"pattern --builder dipoles.invvee:dipole --engine pysim:sinusoidal{O}".split()
    )
