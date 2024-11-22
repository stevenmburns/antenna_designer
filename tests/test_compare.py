from antenna import compare_patterns
from dipole import DipoleBuilder
from invvee import InvVeeBuilder
from hexbeam import HexbeamBuilder, get_hexbeam_data_opt
from bowtie import BowtieBuilder, BowtieArrayBuilder
from moxon import MoxonBuilder

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

def test_compare_hexbeams():
    builder0 = HexbeamBuilder()
    builder1 = HexbeamBuilder(get_hexbeam_data_opt())
    compare_patterns((builder0, builder1), elevation_angle=15)

def test_compare_bowtie_single_vs_array():
    builder0 = BowtieBuilder()
    builder1 = BowtieArrayBuilder()
    compare_patterns((builder0, builder1), elevation_angle=15)

def test_compare_moxons():
    builder0 = MoxonBuilder()
    builder1 = MoxonBuilder(MoxonBuilder.opt_params)
    compare_patterns((builder0, builder1), elevation_angle=15)
