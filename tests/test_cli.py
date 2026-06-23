import antennaknobs as ant

from conftest import needs_pynec

o = " --fn /dev/null"
# o = ''


def test_cli_draw():
    for design in [
        "beams.moxon",
        "verticals.vertical",
        "dipoles.invvee",
        "dipoles.invvee:dipole",
        "arrays.invveearray",
        "specialty.bowtie",
        "arrays.bowtiearray",
        "arrays.bowtiearray2x4",
        "beams.yagi",
        "multiband.fandipole",
    ]:
        ant.cli(f"draw --builder {design}{o}".split())


@needs_pynec
def test_cli_sweep():
    ant.cli(
        f"sweep --param tipspacer_factor --builder beams.moxon --npoints 2{o}".split()
    )
    ant.cli(
        f"sweep --gain --param tipspacer_factor --npoints 2 --builder beams.moxon{o}".split()
    )

    ant.cli(f"sweep --markers 28.57 --npoints 0{o}".split())
    ant.cli(
        f"sweep --markers 28.57 --npoints 0 --builder arrays.invveearray{o}".split()
    )
    ant.cli(
        f"sweep --markers 28.57 --npoints 2 --builder arrays.invveearray{o}".split()
    )
    ant.cli(f"sweep --npoints 2 --builder arrays.invveearray{o}".split())

    ant.cli(f"sweep --markers 28.57 --npoints 0{o} --use_smithchart --z0=50".split())
    ant.cli(
        f"sweep --markers 28.57 --npoints 0 --builder arrays.invveearray{o} --use_smithchart --z0=50".split()
    )
    ant.cli(
        f"sweep --markers 28.57 --npoints 2 --builder arrays.invveearray{o} --use_smithchart --z0=50".split()
    )
    ant.cli(
        f"sweep --npoints 2 --builder arrays.invveearray{o} --use_smithchart --z0=50".split()
    )


@needs_pynec
def test_cli_optimize():
    ant.cli(
        f"optimize --params length_factor angle_radians --builder dipoles.invvee{o}".split()
    )
    ant.cli(
        f"optimize --opt_gain --params length_factor angle_radians --resonance --builder dipoles.invvee{o}".split()
    )


@needs_pynec
def test_cli_pattern():
    ant.cli(f"pattern --builder beams.yagi{o}".split())
    ant.cli(f"pattern --builder dipoles.invvee --wireframe{o}".split())


@needs_pynec
def test_cli_compare_patterns():
    ant.cli(f"compare_patterns{o}".split())
    ant.cli(f"compare_patterns --builders dipoles.invvee beams.moxon{o}".split())
    ant.cli(f"compare_patterns --builders dipoles.invvee beams.hexbeam{o}".split())


def test_cli_engine_flag():
    """--engine momwire selects the momwire backend; --ground forces a
    specific ground model on either engine."""
    dipole = "dipoles.invvee:dipole"
    ant.cli(f"pattern --builder {dipole} --engine momwire --ground free{o}".split())
    ant.cli(f"pattern --builder {dipole} --engine momwire --ground pec{o}".split())
    ant.cli(
        f"compare_patterns --builders {dipole} --engine momwire --ground free{o}".split()
    )
    ant.cli(
        f"sweep --builder {dipole} --npoints 3 --engine momwire --ground free{o}".split()
    )
