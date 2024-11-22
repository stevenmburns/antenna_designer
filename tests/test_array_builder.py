import antenna as ant
from bowtie import BowtieArrayBuilder


def test_newbowtiebuilder():
    b = BowtieArrayBuilder()
    b.build_wires()


def test_bowtie_pattern():
  ant.pattern(BowtieArrayBuilder(), fn='pattern.pdf')

