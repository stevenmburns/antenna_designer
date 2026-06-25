"""Unit tests for web/pynec_backend.py.

Covers the pure helpers and module-level state — `_segment_centers_to_knot_currents`
(averaging seg-centers onto knot positions), module flags, and the
known-error path of `solve()` when an example declares no pynec_solve.

The NEC-driven sweep/pattern paths are not covered here (they need a
running NEC engine; we exercise those via test_momwire_engine.py's
PyNEC comparisons).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from antennaknobs.web import pynec_backend
from antennaknobs.web.examples._base import AntennaExample


def test_have_pynec_flag_is_bool():
    assert isinstance(pynec_backend.HAVE_PYNEC, bool)


def test_module_constants():
    assert pynec_backend.C_LIGHT == 299_792_458.0
    assert pynec_backend.GROUND_DIELECTRIC == 10.0
    assert pynec_backend.GROUND_CONDUCTIVITY == 0.002


def test_segment_centers_average_onto_interior_knots():
    # 3 segs, 4 knots. Interior knots [1] and [2] take averages of
    # adjacent seg-centers; boundaries are zero (open-wire BC).
    seg = np.array([2.0 + 0j, 4.0 + 0j, 8.0 + 0j])
    out = pynec_backend._segment_centers_to_knot_currents(seg, n_knots=4)
    assert out.shape == (4,)
    assert out[0] == 0j
    assert out[1] == 3.0 + 0j
    assert out[2] == 6.0 + 0j
    assert out[3] == 0j


def test_segment_centers_junction_at_start_carries_first_seg_to_boundary():
    seg = np.array([2.0 + 1j, 4.0 + 0j])
    out = pynec_backend._segment_centers_to_knot_currents(
        seg, n_knots=3, junction_at_start=True
    )
    assert out[0] == seg[0]
    assert out[-1] == 0j  # default still zero on the end


def test_segment_centers_junction_at_end_carries_last_seg_to_boundary():
    seg = np.array([2.0 + 0j, 4.0 + 2j])
    out = pynec_backend._segment_centers_to_knot_currents(
        seg, n_knots=3, junction_at_end=True
    )
    assert out[-1] == seg[-1]
    assert out[0] == 0j


def test_segment_centers_length_mismatch_raises():
    # cur_per_seg must be exactly n_knots-1; anything else means the
    # caller mis-routed the wire data.
    seg = np.array([1.0 + 0j, 2.0 + 0j])  # length 2
    with pytest.raises(RuntimeError, match="doesn't match"):
        pynec_backend._segment_centers_to_knot_currents(seg, n_knots=4)


def test_solve_raises_when_example_has_no_pynec_solve():
    # Construct a stub example with pynec_solve=None and inject it.
    # Tests the error branch without needing a real Builder that
    # opts out of PyNEC.
    stub = AntennaExample(
        name="stub_no_pynec",
        label="stub",
        momwire_solve=lambda req: {},
        momwire_sweep=lambda req, freqs: ([], []),
        pynec_solve=None,
    )
    with patch.dict(pynec_backend.EXAMPLES, {"stub_no_pynec": stub}, clear=False):
        with pytest.raises(ValueError, match="PyNEC solve not implemented"):
            pynec_backend.solve({"geometry": "stub_no_pynec"})
