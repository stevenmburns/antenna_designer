"""Tests for AntennaBuilder.draw()."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from antenna_designer import AntennaBuilder


def test_draw_handles_named_edge_designs(tmp_path):
    """draw() used to hard-unpack 4-tuples (p0, p1, _, _), so it raised
    "too many values to unpack" on every named-edge (5-tuple) design -- the
    transmission-line / network builders such as wire.sterba_tl. It
    must render those too, taking the endpoints regardless of tuple arity."""
    from antenna_designer.designs.wire.sterba_tl import Builder

    tups = Builder().build_wires()
    assert any(len(t) == 5 for t in tups)  # this design uses named edges

    out = tmp_path / "sterba_tl.png"
    AntennaBuilder.draw(tups, fn=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_draw_handles_plain_four_tuple_designs(tmp_path):
    """The common 4-tuple case (no named edges) still works."""
    from antenna_designer.designs.beams.moxon import Builder

    out = tmp_path / "moxon.png"
    AntennaBuilder.draw(Builder().build_wires(), fn=str(out))
    assert out.exists() and out.stat().st_size > 0
