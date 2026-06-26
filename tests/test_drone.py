"""Tests for the 3D-turtle Drone and the delta_loop twin built with it."""

import math

import numpy as np
import pytest

from antennaknobs import Drone
from antennaknobs.designs.loops.delta_loop import Builder as DeltaLoop
from antennaknobs.designs.loops.delta_loop_drone import Builder as DeltaLoopDrone
from antennaknobs.designs.loops.horizontal_loop_drone import Builder as HLoopDrone


def test_starts_facing_world_x():
    d = Drone(position=(1.0, 2.0, 3.0))
    assert d.position == pytest.approx((1.0, 2.0, 3.0))
    assert d.heading == pytest.approx((1.0, 0.0, 0.0))


def test_forward_pays_out_one_edge_when_pen_down():
    d = Drone(ref=1.0)
    d.forward(2.0)  # pen up -> moves (turtle penup) but lays no wire
    assert d.wires() == []
    assert d.position == pytest.approx((2.0, 0.0, 0.0))
    d.pay_out().forward(2.0)
    assert len(d.wires()) == 1
    p0, p1, nsegs, ex = d.wires()[0]
    assert p0 == pytest.approx((2.0, 0.0, 0.0))
    assert p1 == pytest.approx((4.0, 0.0, 0.0))
    assert ex is None  # structural


def test_cut_stops_paying_out():
    d = Drone()
    d.pay_out().forward(1.0).cut().forward(1.0)
    assert len(d.wires()) == 1


def test_jump_and_move_to_lay_no_wire():
    d = Drone()
    d.pay_out().jump(5.0)
    d.move_to((0.0, 1.0, 0.0))
    assert d.wires() == []
    assert d.position == pytest.approx((0.0, 1.0, 0.0))


def test_feed_marks_driven_segment():
    d = Drone()
    d.feed(1 + 0j).forward(0.1)
    _, _, _, ex = d.wires()[0]
    assert ex == 1 + 0j


def test_yaw_is_body_relative():
    d = Drone()
    d.yaw(90)
    assert d.heading == pytest.approx((0.0, 1.0, 0.0))  # +x turned to +y
    d.yaw(90)
    assert d.heading == pytest.approx((-1.0, 0.0, 0.0))


def test_equilateral_triangle_closes_to_machine_eps():
    d = Drone(ref=3.0).pay_out()
    for _ in range(3):
        d.forward(3.0)
        d.yaw(120)
    start = np.array(d.wires()[0][0])
    end = np.array(d.wires()[-1][1])
    assert np.abs(start - end).max() < 1e-9


def test_close_flies_home_with_current_pen():
    d = Drone(ref=1.0).pay_out()
    d.forward(1.0).yaw(120).forward(1.0).yaw(120)
    d.feed(1 + 0j).close(nsegs=3)
    last = d.wires()[-1]
    assert last[1] == pytest.approx(d.wires()[0][0])  # back to the origin
    assert last[2] == 3 and last[3] == 1 + 0j


def test_face_sets_heading_and_keeps_position():
    d = Drone(position=(0.0, 0.5, 7.0))
    d.face(heading=(0.0, 1.0, 1.0), up=(1.0, 0.0, 0.0))
    h = np.array(d.heading)
    assert h == pytest.approx(np.array([0.0, 1.0, 1.0]) / math.sqrt(2.0))
    assert d.position == pytest.approx((0.0, 0.5, 7.0))


def test_face_rejects_parallel_up():
    with pytest.raises(ValueError):
        Drone().face(heading=(1.0, 0.0, 0.0), up=(2.0, 0.0, 0.0))


def _key(edges):
    """Edges as an order- and direction-independent multiset (the drone may
    traverse a segment either way)."""
    out = []
    for p0, p1, nsegs, ex in edges:
        a = tuple(round(c, 9) for c in p0)
        b = tuple(round(c, 9) for c in p1)
        out.append((tuple(sorted([a, b])), nsegs, ex))
    return sorted(out)


@pytest.mark.parametrize("variant", ["default", "z100", "z200"])
def test_delta_loop_drone_matches_coordinate_version(variant):
    params = {
        "default": DeltaLoop.default_params,
        "z100": DeltaLoop.z100_params,
        "z200": DeltaLoop.z200_params,
    }[variant]
    # delta_loop_drone only declares default_params, but its build_wires is
    # param-driven, so feed it each delta_loop variant's params directly.
    coord = DeltaLoop(dict(params)).build_wires()
    drone = DeltaLoopDrone(dict(params)).build_wires()
    assert len(drone) == len(coord)
    assert _key(drone) == _key(coord)


def test_horizontal_loop_drone_is_a_closed_planar_square():
    b = HLoopDrone()
    wires = b.build_wires()
    # 5 edges: three full sides, two corner-inset stubs, joined by the
    # diagonal feed across corner A (the last edge, via close()).
    assert len(wires) == 5

    # Flat in the z = base plane.
    base = b.default_params["base"]
    zs = {round(p[2], 9) for e in wires for p in (e[0], e[1])}
    assert zs == {base}

    # Exactly one driven segment, one NEC segment long.
    driven = [e for e in wires if e[3] is not None]
    assert len(driven) == 1
    assert driven[0][2] == 1 and driven[0][3] == 1 + 0j

    # The loop closes exactly, and is a connected walk.
    assert wires[0][0] == pytest.approx(wires[-1][1])
    for prev, nxt in zip(wires, wires[1:]):
        assert prev[1] == pytest.approx(nxt[0])


def test_horizontal_loop_drone_feed_is_symmetric():
    # The feed must sit on a mirror plane of the loop or the pattern skews.
    # Corner A is at (-h, -h); the mirror plane is the A-C diagonal x = y.
    b = HLoopDrone()
    wires = b.build_wires()
    (fx0, fy0, _), (fx1, fy1, _), _, _ = next(e for e in wires if e[3] is not None)

    # The driven segment's two ends are mirror images across x = y...
    assert (fx0, fy0) == pytest.approx((fy1, fx1))
    # ...so its midpoint lies on the diagonal (x == y).
    assert (fx0 + fx1) / 2 == pytest.approx((fy0 + fy1) / 2)

    # And the whole loop is invariant under that reflection (x, y) -> (y, x).
    pts = {(round(p[0], 6), round(p[1], 6)) for e in wires for p in (e[0], e[1])}
    assert {(y, x) for (x, y) in pts} == pts
