"""FastAPI backend for the antenna_designer web UI.

Endpoints (Phase 1):
    GET  /api/builders          — schema for every concrete Builder
    GET  /api/builder/{name}    — schema for one builder + variant
    POST /api/solve             — run a single-frequency solve

Run with:
    python -m uvicorn web.server:app --reload --port 8000
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from antenna_designer.engines.pynec import PyNECEngine
from antenna_designer.engines.pysim import PysimEngine
from antenna_designer.web_schema import builder_schema, list_builders

from pysim import BSplinePySim, SinusoidalPySim, TriangularPySim

app = FastAPI(title="antenna_designer web UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


PYSIM_BASIS = {
    "triangular": TriangularPySim,
    "sinusoidal": SinusoidalPySim,
    "bspline": BSplinePySim,
}


class SolveRequest(BaseModel):
    builder: str = Field(..., description="Dotted builder name, e.g. 'dipole'")
    variant: str = Field("default", description="Variant of default_params")
    params: dict[str, Any] = Field(default_factory=dict, description="Param overrides")
    engine: str = Field("pynec", description="'pynec' or 'pysim'")
    pysim_basis: str = Field("triangular", description="triangular|sinusoidal|bspline")
    ground: str | None = Field(None, description="None|'free'|'pec'|'finite:eps,sigma'")
    far_field: bool = Field(True, description="Include far-field rings in response")


class SweepRequest(BaseModel):
    builder: str
    variant: str = "default"
    params: dict[str, Any] = Field(default_factory=dict)
    engine: str = "pynec"
    pysim_basis: str = "triangular"
    ground: str | None = None
    band_start_mhz: float = Field(..., description="Sweep start (MHz)")
    band_stop_mhz: float = Field(..., description="Sweep stop (MHz)")
    n_points: int = Field(41, ge=2, le=401)


class ConvergeRequest(BaseModel):
    builder: str
    variant: str = "default"
    params: dict[str, Any] = Field(default_factory=dict)
    engine: str = "pynec"
    pysim_basis: str = "triangular"
    ground: str | None = None
    # Scaling factors applied to every wire's n_seg. 1.0 is the builder's
    # native segmentation; sweeping >1 shows whether the solve has
    # converged at the native resolution.
    scales: list[float] = Field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0, 3.0])


def _resolve_builder_class(name: str) -> type:
    try:
        mod = import_module(f"antenna_designer.designs.{name}")
    except ModuleNotFoundError:
        raise HTTPException(404, f"unknown builder: {name}")
    cls = getattr(mod, "Builder", None)
    if cls is None:
        raise HTTPException(404, f"no Builder class in {name}")
    return cls


def _build_builder(req: SolveRequest):
    cls = _resolve_builder_class(req.builder)
    if req.variant and req.variant != "default":
        params_src = getattr(cls, f"{req.variant}_params", None)
        if params_src is None:
            raise HTTPException(400, f"unknown variant: {req.variant}")
        params = dict(params_src)
    else:
        params = dict(cls.default_params)
    # Overlay user-supplied params. Re-hydrate complex params from
    # {"re": .., "im": ..} where the existing default is complex. Keys
    # not declared in default_params (e.g. optional `phase_lr` on array
    # builders) are applied after construction via setattr, matching the
    # existing AntennaBuilder pattern.
    extras: dict[str, Any] = {}
    for k, v in req.params.items():
        if k not in params:
            extras[k] = v
            continue
        if isinstance(params[k], complex) and isinstance(v, dict):
            params[k] = complex(float(v.get("re", 0.0)), float(v.get("im", 0.0)))
        else:
            params[k] = v
    inst = cls(params=params)
    for k, v in extras.items():
        if isinstance(v, dict) and {"re", "im"} <= v.keys():
            v = complex(float(v["re"]), float(v["im"]))
        setattr(inst, k, v)
    return inst


def _parse_ground(s: str | None):
    if s is None or s == "free":
        return None
    if s == "pec":
        return "pec"
    if s.startswith("finite:"):
        try:
            eps_s, sigma_s = s[len("finite:") :].split(",")
            return ("finite", float(eps_s), float(sigma_s))
        except ValueError:
            raise HTTPException(400, f"bad ground spec: {s!r}")
    raise HTTPException(400, f"unknown ground: {s!r}")


def _make_engine(req: SolveRequest, builder):
    ground = _parse_ground(req.ground)
    if req.engine == "pynec":
        # PyNEC's default ground is finite; here None means free space.
        return PyNECEngine(builder, ground=ground if ground is not None else "free")
    if req.engine == "pysim":
        basis = PYSIM_BASIS.get(req.pysim_basis)
        if basis is None:
            raise HTTPException(400, f"unknown pysim basis: {req.pysim_basis}")
        return PysimEngine(builder, solver=basis, ground=ground)
    raise HTTPException(400, f"unknown engine: {req.engine}")


def _pack_wires(builder, wire_currents):
    """Combine builder.build_wires() tuples with engine current data.

    Returns a list of {p0, p1, n_seg, feed_voltage} + a list of
    {knot_positions, knot_currents_re, knot_currents_im} (the latter is
    engine-specific in cardinality)."""
    tups = builder.build_wires()
    wires = [
        {
            "p0": list(map(float, p0)),
            "p1": list(map(float, p1)),
            "n_seg": int(n_seg),
            "feed_voltage": (
                None
                if ev is None
                else {"re": float(np.real(ev)), "im": float(np.imag(ev))}
            ),
        }
        for (p0, p1, n_seg, ev) in tups
    ]
    currents = [
        {
            "knot_positions": w.knot_positions.tolist(),
            "knot_currents_re": w.knot_currents.real.tolist(),
            "knot_currents_im": w.knot_currents.imag.tolist(),
        }
        for w in wire_currents
    ]
    return wires, currents


@app.get("/api/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/builders")
def get_builders():
    return {"builders": list_builders()}


@app.get("/api/builder/{name:path}")
def get_builder(name: str, variant: str = "default"):
    s = builder_schema(name, variant)
    if s is None:
        raise HTTPException(404, f"unknown builder: {name}")
    return s


def _solve_request_from_sweep(req: SweepRequest) -> SolveRequest:
    return SolveRequest(
        builder=req.builder,
        variant=req.variant,
        params=req.params,
        engine=req.engine,
        pysim_basis=req.pysim_basis,
        ground=req.ground,
        far_field=False,
    )


def _scale_builder_segs(builder, scale: float):
    """Monkey-patch builder.build_wires so every wire's n_seg is scaled by
    `scale` (min 1). AntennaBuilder.__setattr__ diverts plain attribute
    writes into `_params`, so we install the wrapper via `__dict__` to
    bypass that and shadow the class-level method on this instance.
    The engine constructors call build_wires() once, so one patch per
    builder instance suffices."""
    orig = builder.build_wires

    def wrapped():
        return [
            (p0, p1, max(1, int(round(n * scale))), ev) for (p0, p1, n, ev) in orig()
        ]

    builder.__dict__["build_wires"] = wrapped
    return builder


@app.post("/api/converge")
def converge(req: ConvergeRequest):
    """Re-solve at increasing segmentations to gauge whether the live
    solve is past its convergence knee."""
    solve_req = SolveRequest(
        builder=req.builder,
        variant=req.variant,
        params=req.params,
        engine=req.engine,
        pysim_basis=req.pysim_basis,
        ground=req.ground,
        far_field=False,
    )
    n_segs_total: list[int] = []
    z_per_feed: list[list[dict]] = []
    for s in req.scales:
        b = _build_builder(solve_req)
        _scale_builder_segs(b, s)
        eng = _make_engine(solve_req, b)
        zs = eng.impedance()
        n_segs_total.append(sum(t[2] for t in b.build_wires()))
        z_per_feed.append([{"re": float(z.real), "im": float(z.imag)} for z in zs])
    return {
        "scales": list(map(float, req.scales)),
        "n_segs_total": n_segs_total,
        "z_per_feed": z_per_feed,
    }


@app.post("/api/sweep")
def sweep(req: SweepRequest):
    """Run an evenly-spaced frequency sweep on the chosen engine.

    PyNEC's batched sweep card requires evenly-spaced frequencies; pysim
    handles arbitrary spacing but we send evenly-spaced for parity. The
    builder's `freq` attribute is only used for k-independent setup; the
    actual sweep targets are passed in via impedance_sweep(freqs)."""
    builder = _build_builder(_solve_request_from_sweep(req))
    eng = _make_engine(_solve_request_from_sweep(req), builder)
    freqs = np.linspace(req.band_start_mhz, req.band_stop_mhz, req.n_points)
    zs = eng.impedance_sweep(freqs)  # (n_freqs, n_feeds)
    return {
        "freqs_mhz": freqs.tolist(),
        "z_per_feed": [
            [{"re": float(z.real), "im": float(z.imag)} for z in row] for row in zs
        ],
    }


@app.post("/api/solve")
def solve(req: SolveRequest):
    builder = _build_builder(req)
    eng = _make_engine(req, builder)

    z_per_feed = [{"re": float(z.real), "im": float(z.imag)} for z in eng.impedance()]
    currents = eng.current_distribution()
    wires_json, currents_json = _pack_wires(builder, currents)

    out: dict[str, Any] = {
        "builder": req.builder,
        "variant": req.variant,
        "engine": req.engine,
        "freq_mhz": float(builder.freq),
        "z_per_feed": z_per_feed,
        "wires": wires_json,
        "currents": currents_json,
    }

    if req.far_field:
        try:
            ff = eng.far_field(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
            out["far_field"] = {
                "thetas": ff.thetas.tolist(),
                "phis": ff.phis.tolist(),
                "rings": ff.rings if isinstance(ff.rings, list) else ff.rings.tolist(),
                "max_gain": float(ff.max_gain),
                "min_gain": float(ff.min_gain),
            }
        except NotImplementedError:
            out["far_field"] = None

    return out
