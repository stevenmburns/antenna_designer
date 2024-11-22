from antenna import compare_patterns
from dipole import DipoleBuilder, get_dipole_data
from invvee import InvVeeBuilder, get_invvee_data
from hexbeam import HexbeamBuilder, get_hexbeam_data, get_hexbeam_data_opt
from bowtie import BowtieBuilder, get_bowtie_data, BowtieSingleBuilder, get_single_bowtie_data

def test_compare():
    builder0 = DipoleBuilder(**get_dipole_data())
    builder1 = InvVeeBuilder(**get_invvee_data())
    builder2 = BowtieBuilder(**get_bowtie_data())
    builder3 = BowtieSingleBuilder(**get_single_bowtie_data())
    builder4 = HexbeamBuilder(**get_hexbeam_data_opt())
    compare_patterns((builder0, builder1, builder2, builder3, builder4), elevation_angle=15)

def test_compare_hexbeams():
    builder0 = HexbeamBuilder(**get_hexbeam_data())
    builder1 = HexbeamBuilder(**get_hexbeam_data_opt())
    compare_patterns((builder0, builder1), elevation_angle=15)
