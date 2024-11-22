from antenna import compare_patterns
from dipole import DipoleBuilder
from invvee import InvVeeBuilder
from hexbeam import HexbeamBuilder, get_hexbeam_data_opt
from bowtie import BowtieBuilder, BowtieSingleBuilder

def test_compare():
    builder0 = DipoleBuilder()
    builder1 = InvVeeBuilder()
    builder2 = BowtieBuilder()
    builder3 = BowtieSingleBuilder()
    builder4 = HexbeamBuilder(get_hexbeam_data_opt())
    compare_patterns((builder0, builder1, builder2, builder3, builder4), elevation_angle=15)

def test_compare_hexbeams():
    builder0 = HexbeamBuilder()
    builder1 = HexbeamBuilder(get_hexbeam_data_opt())
    compare_patterns((builder0, builder1), elevation_angle=15)

def test_compare_bowtie_single_vs_array():
    builder0 = BowtieSingleBuilder()
    builder1 = BowtieBuilder()
    compare_patterns((builder0, builder1), elevation_angle=15)
