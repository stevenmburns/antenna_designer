import pytest
from antenna_designer import compare_patterns
from antenna_designer.designs.dipole import DipoleBuilder
from antenna_designer.designs.invvee import InvVeeBuilder, InvveeArrayBuilder
from antenna_designer.designs.hexbeam import HexbeamBuilder, get_hexbeam_data_opt
from antenna_designer.designs.bowtie import BowtieBuilder, BowtieArrayBuilder
from antenna_designer.designs.moxon import MoxonBuilder

@pytest.mark.skip(reason="Draws to screen")
def test_compare():
    builders = (
        DipoleBuilder(),
        InvVeeBuilder(),
        BowtieBuilder(),
        BowtieArrayBuilder(),
        HexbeamBuilder(get_hexbeam_data_opt()),
        MoxonBuilder(MoxonBuilder.opt_params)
    )
    compare_patterns(builders, elevation_angle=15)

@pytest.mark.skip(reason="Draws to screen")
def test_compare_hexbeams():
    builder0 = HexbeamBuilder()
    builder1 = HexbeamBuilder(get_hexbeam_data_opt())
    compare_patterns((builder0, builder1), elevation_angle=15)

@pytest.mark.skip(reason="Draws to screen")
def test_compare_bowtie_single_vs_array():
    builder0 = BowtieBuilder()
    builder1 = BowtieArrayBuilder()
    compare_patterns((builder0, builder1), elevation_angle=15)

@pytest.mark.skip(reason="Draws to screen")
def test_compare_bowtie_invvee_arrays():
    builder0 = InvveeArrayBuilder()
    builder1 = BowtieArrayBuilder()
    compare_patterns((builder0, builder1), elevation_angle=15)

@pytest.mark.skip(reason="Draws to screen")
def test_compare_moxons():
    builder0 = MoxonBuilder()
    builder1 = MoxonBuilder(MoxonBuilder.opt_params)
    compare_patterns((builder0, builder1), elevation_angle=15)
