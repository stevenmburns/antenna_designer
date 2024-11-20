from antenna import *
from dipole import *
from invvee import *
from bowtie import *

def test_compare():
    builder0 = DipoleBuilder(**get_dipole_data())
    builder1 = InvVeeBuilder(**get_invvee_data())
    builder2 = BowtieBuilder(**get_bowtie_data())
    compare_patterns((builder0, builder1, builder2), elevation_angle=5)
    #compare_patterns((builder1, builder2))
