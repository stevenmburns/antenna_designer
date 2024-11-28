import pytest
import numpy as np
from antenna_designer import compare_patterns
from antenna_designer.designs import dipole, invvee, invveearray, hexbeam, bowtie, bowtiearray, moxon

@pytest.mark.skip(reason="Draws to screen")
def test_compare():
    builders = (
        dipole.Builder(),
        invvee.Builder(),
        bowtie.Builder(),
        bowtiearray.Builder(),
        hexbeam.Builder(hexbeam.Builder.opt_params),
        moxon.Builder(moxon.Builder.opt_params)
    )
    compare_patterns(builders, elevation_angle=15)

@pytest.mark.skip(reason="Draws to screen")
def test_compare_excitations():
    angles = np.linspace(0, np.pi/2, 5)
    phases = np.exp((0+1j)*angles)
    builders = (
      invvee.Builder(dict(invvee.Builder.default_params, excitation=ex))
        for ex in phases
    )
    compare_patterns(builders, elevation_angle=15)

@pytest.mark.skip(reason="Draws to screen")
def test_compare_hexbeams():
    builder0 = hexbeam.Builder()
    builder1 = hexbeam.Builder(hexbeam.Builder.opt_params)
    compare_patterns((builder0, builder1), elevation_angle=15)

@pytest.mark.skip(reason="Draws to screen")
def test_compare_bowtie_single_vs_array():
    builder0 = bowtie.Builder()
    builder1 = bowtiearray.Builder()
    compare_patterns((builder0, builder1), elevation_angle=15)

@pytest.mark.skip(reason="Draws to screen")
def test_compare_bowtie_invvee_arrays():
    builder0 = invveearray.Builder()
    builder1 = bowtiearray.Builder()
    compare_patterns((builder0, builder1), elevation_angle=15)

@pytest.mark.skip(reason="Draws to screen")
def test_compare_moxons():
    builder0 = moxon.Builder()
    builder1 = moxon.Builder(moxon.Builder.opt_params)
    compare_patterns((builder0, builder1), elevation_angle=15)
