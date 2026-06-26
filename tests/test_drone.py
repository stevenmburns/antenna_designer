"""Tests for the 3D-turtle Drone and the delta_loop twin built with it."""

import math

import numpy as np
import pytest

from antennaknobs import Drone
from antennaknobs.designs.loops.delta_loop import Builder as DeltaLoop
from antennaknobs.designs.loops.delta_loop_drone import Builder as DeltaLoopDrone


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
