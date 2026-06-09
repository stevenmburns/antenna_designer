"""Unit tests for web/server.py.

Covers the pure helpers (no FastAPI), the JSON-shape contracts the
frontend depends on (/healthz, /examples), and one end-to-end /solve
through the lightest geometry (dipole) so the request → response
pipeline is exercised without dragging in expensive sweeps.

The expensive endpoints (/sweep, /converge, /pattern, /ws) are
streaming/async and are deliberately *not* covered here — those are
integration territory and want their own targeted tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from web import server


# ---------------------------------------------------------------------------
# Test client — shared across the whole module so FastAPI's startup runs once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_physical_cpu_count_is_positive():
    assert server._physical_cpu_count() >= 1


def test_read_ground_off_returns_zero_offset():
    on, h, z = server._read_ground({})
    assert on is False
    assert h == 0.0
    assert z == 0.0


def test_read_ground_on_sets_z_offset_to_height():
    on, h, z = server._read_ground({"ground": True, "height_m": 4.5})
    assert on is True
    assert h == 4.5
    assert z == 4.5


def test_read_ground_off_ignores_height():
    # An explicit height with ground=False shouldn't displace the antenna —
    # the geometry stays at its native z=0. The frontend toggles ground
    # independently of the height slider.
    _on, _h, z = server._read_ground({"ground": False, "height_m": 7.0})
    assert z == 0.0


def test_polyline_knots_dedup_shared_corners():
    poly = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    knots = server._polyline_knots(poly, [2, 3])
    # 2 + 3 segments with shared mid-corner → 2 + 3 + 1 = 6 knots, not 7.
    assert knots.shape == (6, 3)
    # First knot of segment 2 is the last knot of segment 1, not duplicated.
    np.testing.assert_allclose(knots[2], poly[1])


def test_sample_arc_for_wire_interleaves_knots_and_midpoints():
    # Wire: three colinear knots at x = 0, 1, 3. h_seg = [1, 2].
    # arc_at_knot = [0, 1, 3]; mid_arc = [0.5, 2.0]
    # sample_arc = [0, 0.5, 1, 2.0, 3]
    knots = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    arc = server._sample_arc_for_wire(knots)
    np.testing.assert_allclose(arc, [0.0, 0.5, 1.0, 2.0, 3.0])


def test_attach_derived_em_fields_computes_wavenumber():
    # k = 2π f / c at the frontend's reference c (299_792_458 m/s).
    f_mhz = 30.0
    out = {"measurement_freq_mhz": f_mhz, "ground": False}
    server._attach_derived_em_fields(out)
    expected_k = 2 * np.pi * f_mhz * 1e6 / server.C_LIGHT
    assert out["k_meas_m_inv"] == pytest.approx(expected_k, rel=1e-12)
    # σ=0 → imaginary permittivity component is exactly zero.
    assert out["ground_eps_im"] == 0.0


def test_attach_derived_em_fields_ground_sigma_negates_into_eps_im():
    # With σ > 0, ground_eps_im = -σ / (ω ε₀) < 0.
    out = {
        "measurement_freq_mhz": 14.0,
        "ground": True,
        "ground_sigma": 0.005,
    }
    server._attach_derived_em_fields(out)
    assert out["ground_eps_im"] < 0.0


# ---------------------------------------------------------------------------
# /healthz — the smoke test the dev launcher polls
# ---------------------------------------------------------------------------


def test_healthz_returns_ok(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ---------------------------------------------------------------------------
# /examples — schema serialization, used by the frontend on mount
# ---------------------------------------------------------------------------


def test_examples_endpoint_returns_every_registered_example(client: TestClient):
    payload = client.get("/examples").json()
    assert "examples" in payload
    names = {e["name"] for e in payload["examples"]}
    # Every registered geometry shows up.
    assert names == set(server.EXAMPLES.keys())


def test_examples_are_sorted_by_label(client: TestClient):
    payload = client.get("/examples").json()
    labels = [e["label"] for e in payload["examples"]]
    assert labels == sorted(labels)


def test_each_example_has_the_keys_the_frontend_reads(client: TestClient):
    payload = client.get("/examples").json()
    required = {
        "name",
        "label",
        "multi_feed",
        "param_schema",
        "result_schema",
        "bands",
        "meas_freq_range_mhz",
        "default_view",
        "default_freq_mhz",
        "has_design_freq",
        "variants",
        "variant_values",
        "sweep_policy",
    }
    for ex in payload["examples"]:
        missing = required - set(ex.keys())
        assert not missing, f"{ex['name']}: missing keys {missing}"


def test_examples_serialize_param_groups_with_kind_group(client: TestClient):
    # fandipole is the canonical group-bearing geometry — its `bands`
    # ParamGroupSpec must round-trip with kind="group" so the frontend's
    # generic schema renderer knows to draw a repeating section.
    payload = client.get("/examples").json()
    fan = next(e for e in payload["examples"] if e["name"] == "freq_based.fandipole")
    groups = [p for p in fan["param_schema"] if p.get("kind") == "group"]
    assert groups, "fandipole bands group missing from serialized schema"
    g = groups[0]
    assert g["name"] == "bands"
    assert g["repeat_count"] == "n_bands"
    assert g["max_repeats"] == 5
    # Inner params are serialized as a list of ParamSpec dicts.
    assert {p["name"] for p in g["params"]} == {"freq", "length_factor"}


def test_examples_carry_default_view_in_valid_set(client: TestClient):
    payload = client.get("/examples").json()
    for ex in payload["examples"]:
        assert ex["default_view"] in {"xy", "yz", "xz"}


def test_examples_carry_sweep_policy_keys(client: TestClient):
    payload = client.get("/examples").json()
    for ex in payload["examples"]:
        sp = ex["sweep_policy"]
        assert set(sp) == {"anchor", "lo_factor", "hi_factor", "band_locked"}
        assert sp["anchor"] in {"design_freq", "meas_freq"}


# ---------------------------------------------------------------------------
# solve() dispatcher — exercise the pysim path end-to-end on the cheapest
# geometry (dipole). This is the only place the test module calls a real
# solver; everything else stays I/O-only.
# ---------------------------------------------------------------------------


def test_solve_dispatches_to_pysim_for_dipole():
    out = server.solve(
        {
            "geometry": "dipole",
            "measurement_freq_mhz": 14.0,
            "design_freq_mhz": 14.0,
            "pysim_model": "triangular",
        }
    )
    assert out["solver"] == "pysim"
    assert out["geometry"] == "dipole"
    # _attach_derived_em_fields ran.
    assert "k_meas_m_inv" in out
    assert out["k_meas_m_inv"] > 0
    # _compute_directivity_norm ran.
    assert "directivity_norm" in out
    assert out["directivity_norm"] > 0
    # Real dipole has a real-part impedance roughly in the tens of ohms.
    assert out["z_in_re"] > 0


def test_solve_falls_back_when_geometry_unknown():
    # An unknown geometry should silently fall back to the first registered
    # example rather than 500 — the frontend can briefly send a stale name
    # while it reloads /examples.
    out = server.solve(
        {
            "geometry": "this_geometry_does_not_exist",
            "measurement_freq_mhz": 14.0,
        }
    )
    assert out["solver"] == "pysim"
    assert "wires" in out


# ---------------------------------------------------------------------------
# _wire_record — packs one wire's knot/sample currents for the JSON response.
# Pure data wrangling, no solver involvement.
# ---------------------------------------------------------------------------


def test_wire_record_packs_knot_data_without_samples():
    knots = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    currents = np.array([0.5 + 0j, 1.0 + 0.2j, 0.0 + 0j])
    out = server._wire_record(knots, currents, label="wire7")
    assert out["label"] == "wire7"
    assert out["knot_positions"] == knots.tolist()
    assert out["knot_currents_re"] == [0.5, 1.0, 0.0]
    assert out["knot_currents_im"] == [0.0, 0.2, 0.0]
    # No sample keys when sample_currents is omitted.
    assert "sample_positions" not in out


def test_wire_record_packs_sample_data_when_provided():
    knots = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    currents = np.array([0.0 + 0j, 1.0 + 0j, 0.0 + 0j])
    # 2 segments → 2*2+1 = 5 sample currents required.
    samples = np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=complex)
    out = server._wire_record(knots, currents, "w", sample_currents=samples)
    # Interleaved positions: knot, midpoint, knot, midpoint, knot.
    pos = np.asarray(out["sample_positions"])
    assert pos.shape == (5, 3)
    np.testing.assert_allclose(pos[0], knots[0])
    np.testing.assert_allclose(pos[2], knots[1])
    np.testing.assert_allclose(pos[4], knots[2])
    np.testing.assert_allclose(pos[1], 0.5 * (knots[0] + knots[1]))
    assert out["sample_currents_re"] == [0.0, 0.5, 1.0, 0.5, 0.0]


def test_wire_record_rejects_currents_length_mismatch():
    knots = np.zeros((3, 3))
    currents = np.zeros(4, dtype=complex)  # one too many
    with pytest.raises(ValueError, match="currents/knots length mismatch"):
        server._wire_record(knots, currents, "w")


def test_wire_record_rejects_sample_currents_length_mismatch():
    knots = np.zeros((3, 3))
    currents = np.zeros(3, dtype=complex)
    bad_samples = np.zeros(4, dtype=complex)  # need 2*2+1 = 5
    with pytest.raises(ValueError, match="sample_currents length"):
        server._wire_record(knots, currents, "w", sample_currents=bad_samples)


# ---------------------------------------------------------------------------
# _compute_directivity_norm — pure-numpy integration over a synthetic
# wire grid. Doesn't need a solver; we feed it a hand-built response dict.
# ---------------------------------------------------------------------------


def _hertzian_dipole_response(freq_mhz: float = 30.0):
    """A 0.1 m centred-z wire with unit current — small enough that the
    radiation integral degenerates to a Hertzian-dipole pattern, large
    enough to keep the numerics well-conditioned.
    """
    knots = np.array(
        [[0.0, 0.0, -0.05], [0.0, 0.0, 0.0], [0.0, 0.0, 0.05]], dtype=float
    )
    return {
        "measurement_freq_mhz": freq_mhz,
        "ground": False,
        "wires": [
            {
                "knot_positions": knots.tolist(),
                "knot_currents_re": [1.0, 1.0, 1.0],
                "knot_currents_im": [0.0, 0.0, 0.0],
            }
        ],
    }


def test_compute_directivity_norm_positive_no_ground():
    out = _hertzian_dipole_response()
    server._attach_derived_em_fields(out)
    server._compute_directivity_norm(out, n_theta=15, n_phi=30)
    assert "directivity_norm" in out
    # ∫|M_perp|² dΩ > 0 for a non-zero current, so the norm is finite +.
    assert out["directivity_norm"] > 0
    assert np.isfinite(out["directivity_norm"])


# ---------------------------------------------------------------------------
# Streaming endpoints + /pattern dispatcher — TestClient drives the routes
# against the lightest geometry (dipole) so each call stays sub-second.
# ---------------------------------------------------------------------------


def _ndjson_records(response_text: str) -> list[dict]:
    return [
        __import__("json").loads(line)
        for line in response_text.splitlines()
        if line.strip()
    ]


def test_sweep_endpoint_empty_freqs_returns_only_done(client: TestClient):
    r = client.post("/sweep", json={"geometry": "dipole", "freqs_mhz": []})
    assert r.status_code == 200
    recs = _ndjson_records(r.text)
    assert recs == [{"done": True, "solver": "pysim"}]


def test_sweep_endpoint_streams_one_record_per_freq_then_done(client: TestClient):
    freqs = [13.5, 14.0, 14.5]
    r = client.post(
        "/sweep",
        json={
            "geometry": "dipole",
            "freqs_mhz": freqs,
            "measurement_freq_mhz": 14.0,
            "pysim_model": "triangular",
        },
    )
    assert r.status_code == 200
    recs = _ndjson_records(r.text)
    assert len(recs) == len(freqs) + 1
    *points, terminator = recs
    assert terminator == {"done": True, "solver": "pysim"}
    for f, rec in zip(freqs, points):
        assert rec["freq_mhz"] == f
        assert rec["solver"] == "pysim"
        assert isinstance(rec["z_re"], float)
        assert isinstance(rec["z_im"], float)
        # Real dipole impedance never goes pathologically far from order
        # ~50 Ω over a ±10% sweep — guards against a future signed-units
        # regression that produced -j×j-style mixups.
        assert abs(rec["z_re"]) < 1e4
        assert abs(rec["z_im"]) < 1e4


def test_sweep_endpoint_returns_ndjson_content_type(client: TestClient):
    r = client.post("/sweep", json={"geometry": "dipole", "freqs_mhz": []})
    assert r.headers["content-type"].startswith("application/x-ndjson")


def test_converge_endpoint_streams_one_record_per_n_then_done(client: TestClient):
    ns = [3, 5]
    r = client.post(
        "/converge",
        json={
            "geometry": "dipole",
            "n_values": ns,
            "measurement_freq_mhz": 14.0,
            "pysim_model": "triangular",
        },
    )
    assert r.status_code == 200
    recs = _ndjson_records(r.text)
    assert len(recs) == len(ns) + 1
    *points, terminator = recs
    assert terminator == {"done": True, "solver": "pysim"}
    for n, rec in zip(ns, points):
        assert rec["n_per_wire"] == n
        assert rec["solver"] == "pysim"
        # Convergence trace should always carry real impedance fields —
        # the error-path branch yields an `error` key instead, which
        # would mean dipole crapped out at this N (it shouldn't).
        assert "z_re" in rec and "z_im" in rec


def test_converge_endpoint_empty_n_values_returns_only_done(client: TestClient):
    r = client.post("/converge", json={"geometry": "dipole", "n_values": []})
    recs = _ndjson_records(r.text)
    assert recs == [{"done": True, "solver": "pysim"}]


def test_pattern_endpoint_pysim_returns_unavailable(client: TestClient):
    # /pattern is PyNEC-only; pysim solvers get the {"available": False}
    # short-circuit. Tests both the solver-flag path and the
    # HAVE_PYNEC=False fallback shape.
    r = client.post(
        "/pattern",
        json={"geometry": "dipole", "solver": "pysim", "measurement_freq_mhz": 14.0},
    )
    assert r.status_code == 200
    assert r.json() == {"available": False}


# ---------------------------------------------------------------------------
# _solve_z_only — pure helper inlined from converge's per-point loop.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PyNEC paths — same endpoints, solver="pynec" branch. Each test runs one
# real NEC solve on dipole; sub-second on the CI box.
# ---------------------------------------------------------------------------


# Skip the whole PyNEC-path section when PyNEC didn't build (it's a hard
# dep on CI, but local devs without swig+gfortran shouldn't see a hard
# failure). pynec_backend.HAVE_PYNEC is the same flag the dispatcher uses.
pynec_required = pytest.mark.skipif(
    not __import__("web.pynec_backend", fromlist=["HAVE_PYNEC"]).HAVE_PYNEC,
    reason="PyNEC not built in this environment",
)


@pynec_required
def test_solve_dispatches_to_pynec_when_requested():
    out = server.solve(
        {
            "geometry": "dipole",
            "solver": "pynec",
            "measurement_freq_mhz": 14.0,
        }
    )
    assert out["geometry"] == "dipole"
    assert "wires" in out
    # _attach_derived_em_fields + _compute_directivity_norm wrap both
    # solver branches; their outputs must show up regardless of which
    # backend ran.
    assert out["k_meas_m_inv"] > 0
    assert out["directivity_norm"] > 0
    # Real dipole at 14 MHz: real Z roughly order 73 Ω; very loose bound
    # so a future PyNEC version bump doesn't trip the test.
    assert 10 < out["z_in_re"] < 500


@pynec_required
def test_sweep_endpoint_with_pynec_streams_per_point(client: TestClient):
    freqs = [13.8, 14.0, 14.2]
    r = client.post(
        "/sweep",
        json={
            "geometry": "dipole",
            "solver": "pynec",
            "freqs_mhz": freqs,
            "measurement_freq_mhz": 14.0,
        },
    )
    assert r.status_code == 200
    recs = _ndjson_records(r.text)
    assert len(recs) == len(freqs) + 1
    *points, terminator = recs
    assert terminator == {"done": True, "solver": "pynec"}
    for f, rec in zip(freqs, points):
        assert rec["freq_mhz"] == f
        assert rec["solver"] == "pynec"
        assert isinstance(rec["z_re"], float) and isinstance(rec["z_im"], float)


@pynec_required
def test_pattern_endpoint_with_pynec_returns_full_grid(client: TestClient):
    # /pattern routes through pynec_backend.pattern → rp_card → gain
    # extraction. Asserts the grid shape the frontend reads (46 thetas
    # × 73 phis) is what comes back, and that all gains are finite
    # (NaN/Inf would imply a botched ex_card or fr_card sequence).
    r = client.post(
        "/pattern",
        json={
            "geometry": "dipole",
            "solver": "pynec",
            "measurement_freq_mhz": 14.0,
        },
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["available"] is True
    assert payload["geometry"] == "dipole"
    assert len(payload["theta_deg"]) == 46
    assert len(payload["phi_deg"]) == 73
    assert len(payload["gain_dbi"]) == 46
    assert len(payload["gain_dbi"][0]) == 73
    # NEC reports -999.99 dBi at radiation nulls as a sentinel — filter
    # those before bounds-checking. Anything outside (-200, 30) on the
    # non-null cells implies a malformed solve, not a real antenna.
    flat = [g for row in payload["gain_dbi"] for g in row]
    real_gains = [g for g in flat if g > -900]
    assert real_gains, "NEC returned no non-null gain cells"
    assert all(-200 < g < 30 for g in real_gains)
    # Peak gain for a half-wave dipole is ~2.15 dBi (in free space).
    # Loose ceiling guards against accidental dBW vs dBi mixups.
    assert max(real_gains) < 10


@pynec_required
def test_sweep_endpoint_pynec_empty_freqs_returns_only_done(client: TestClient):
    r = client.post(
        "/sweep", json={"geometry": "dipole", "solver": "pynec", "freqs_mhz": []}
    )
    recs = _ndjson_records(r.text)
    assert recs == [{"done": True, "solver": "pynec"}]


# ---------------------------------------------------------------------------
# Multi-feed dispatch — adapter._auto_multi_feed sets multi_feed=True
# whenever a design's build_wires() declares >1 excitation. /solve and
# /sweep both light up the per-feed response shape for those designs.
# ---------------------------------------------------------------------------


def test_multi_feed_flag_lights_up_for_array_designs():
    # 16 designs in the registry have >1 feed wire; all should be flagged
    # after the auto-detect lands. Pin a few canonical names so a future
    # refactor that drops the auto-detect path gets caught.
    ex = server.EXAMPLES["bowtiearray1x2"]
    assert ex.multi_feed is True
    assert server.EXAMPLES["invveearray"].multi_feed is True
    assert server.EXAMPLES["dipole"].multi_feed is False


def test_solve_for_multi_feed_geometry_includes_feeds_array():
    out = server.solve(
        {
            "geometry": "bowtiearray1x2",
            "measurement_freq_mhz": 28.5,
            "pysim_model": "triangular",
        }
    )
    assert "feeds" in out
    assert len(out["feeds"]) == 2  # bowtiearray1x2 has two driven elements
    for f in out["feeds"]:
        assert set(f) == {"z_re", "z_im", "v_re", "v_im"}
        assert isinstance(f["z_re"], float)
        assert isinstance(f["v_re"], float)
    # Primary z_in_re must match feeds[0].z_re — the primary impedance
    # field has always been a duplicate of the first feed.
    assert out["z_in_re"] == pytest.approx(out["feeds"][0]["z_re"])
    assert out["z_in_im"] == pytest.approx(out["feeds"][0]["z_im"])


def test_solve_for_single_feed_geometry_omits_feeds_array():
    out = server.solve(
        {
            "geometry": "dipole",
            "measurement_freq_mhz": 14.0,
            "pysim_model": "triangular",
        }
    )
    assert "feeds" not in out


def test_phase_param_slider_range_spans_full_unit_circle():
    # phase_lr / phase_tb default to 0.0; without the adapter's
    # phase_*-name special case the auto-derive falls back to (-1, 1)
    # for default=0, which is a useless 2° span. Confirm the unit-circle
    # default kicks in instead.
    import importlib

    schema = importlib.import_module(
        "antenna_designer.designs.bowtiearray"
    ).Builder.default_params
    assert "phase_lr" in schema and "phase_tb" in schema
    from web.examples import REGISTRY

    by_name = {s.name: s for s in REGISTRY["bowtiearray"].param_schema}
    lr = by_name["phase_lr"]
    assert (lr.min, lr.max, lr.step) == (-180.0, 180.0, 1.0)
    assert lr.unit == "°"
    assert lr.precision == 0


def test_phase_lr_drives_per_feed_voltage_phasor():
    # bowtiearray1x2 already tests this through flat_wires_to_polylines;
    # this is the end-to-end check that /solve's feeds[i].v_re/v_im
    # actually reflect the phase_lr setting.
    out_zero = server.solve(
        {
            "geometry": "bowtiearray1x2",
            "measurement_freq_mhz": 28.5,
            "pysim_model": "triangular",
            "phase_lr": 0.0,
        }
    )
    out_quad = server.solve(
        {
            "geometry": "bowtiearray1x2",
            "measurement_freq_mhz": 28.5,
            "pysim_model": "triangular",
            "phase_lr": 90.0,
        }
    )
    # phase_lr=0: both feed voltages are real (V0 = V1 = 1+0j).
    f0_zero, f1_zero = out_zero["feeds"]
    assert f0_zero["v_re"] == pytest.approx(1.0)
    assert f0_zero["v_im"] == pytest.approx(0.0)
    assert f1_zero["v_re"] == pytest.approx(1.0)
    assert f1_zero["v_im"] == pytest.approx(0.0)
    # phase_lr=90: V0 = 1+0j, V1 = j (0 + 1j).
    f0_q, f1_q = out_quad["feeds"]
    assert f0_q["v_re"] == pytest.approx(1.0)
    assert f0_q["v_im"] == pytest.approx(0.0)
    assert f1_q["v_re"] == pytest.approx(0.0, abs=1e-10)
    assert f1_q["v_im"] == pytest.approx(1.0)


def test_sweep_endpoint_streams_feeds_z_for_multi_feed_geometry(client: TestClient):
    r = client.post(
        "/sweep",
        json={
            "geometry": "bowtiearray1x2",
            "freqs_mhz": [28.4, 28.5],
            "pysim_model": "triangular",
        },
    )
    assert r.status_code == 200
    recs = _ndjson_records(r.text)
    *points, terminator = recs
    assert terminator == {"done": True, "solver": "pysim"}
    for rec in points:
        assert "feeds_z_re" in rec
        assert "feeds_z_im" in rec
        assert len(rec["feeds_z_re"]) == 2
        assert len(rec["feeds_z_im"]) == 2
        # Primary z must mirror feed 0 — same invariant as /solve.
        assert rec["z_re"] == pytest.approx(rec["feeds_z_re"][0])


# ---------------------------------------------------------------------------
# /ws — websocket endpoint. TestClient.websocket_connect gives a synchronous
# context manager around the live route.
# ---------------------------------------------------------------------------


def test_ws_endpoint_round_trips_a_solve(client: TestClient):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(
            __import__("json").dumps(
                {
                    "geometry": "dipole",
                    "measurement_freq_mhz": 14.0,
                    "pysim_model": "triangular",
                }
            )
        )
        result = __import__("json").loads(ws.receive_text())
    assert result["solver"] == "pysim"
    assert result["geometry"] == "dipole"
    assert "wires" in result
    assert result["z_in_re"] > 0


def test_ws_endpoint_handles_multiple_requests_on_one_socket(client: TestClient):
    # The endpoint's outer `while True` loop must keep serving once the
    # first solve returns; the React frontend reuses the same socket
    # for every slider drag.
    req = __import__("json").dumps(
        {
            "geometry": "dipole",
            "measurement_freq_mhz": 14.0,
            "pysim_model": "triangular",
        }
    )
    with client.websocket_connect("/ws") as ws:
        ws.send_text(req)
        first = __import__("json").loads(ws.receive_text())
        ws.send_text(req)
        second = __import__("json").loads(ws.receive_text())
    assert first["geometry"] == "dipole"
    assert second["geometry"] == "dipole"
    # Same request → deterministic same z_in.
    assert first["z_in_re"] == pytest.approx(second["z_in_re"])
    assert first["z_in_im"] == pytest.approx(second["z_in_im"])


def test_ws_endpoint_returns_cleanly_on_client_disconnect(client: TestClient):
    # Opening + closing the socket without sending anything has to hit
    # the outer WebSocketDisconnect path (receive_text raises). The
    # endpoint must catch it cleanly — no exception leaking out of the
    # context manager.
    with client.websocket_connect("/ws"):
        pass  # context exit closes the socket


def test_solve_z_only_returns_primary_z_and_no_feeds_for_dipole():
    z, feeds_z = server._solve_z_only(
        {
            "geometry": "dipole",
            "measurement_freq_mhz": 14.0,
            "pysim_model": "triangular",
        }
    )
    assert isinstance(z, complex)
    assert z.real > 0  # real-input dipole has positive real Z
    assert feeds_z is None  # single-feed


def test_compute_directivity_norm_ground_on_stays_finite_and_positive():
    # With ground=True the integration domain halves and a reflected
    # image contribution is added; the resulting norm has to stay finite
    # and positive. (The exact ground/no-ground ratio depends on source
    # geometry and isn't a clean closed form.)
    with_ground = _hertzian_dipole_response()
    with_ground["ground"] = True
    # PEC ground: real Fresnel code path takes complex eps_r + j*eps_im.
    with_ground["ground_eps_r"] = 1.0e10
    with_ground["ground_sigma"] = 0.0
    server._attach_derived_em_fields(with_ground)
    server._compute_directivity_norm(with_ground, n_theta=15, n_phi=30)
    assert with_ground["directivity_norm"] > 0
    assert np.isfinite(with_ground["directivity_norm"])
