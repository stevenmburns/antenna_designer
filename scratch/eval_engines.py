"""Cross-engine evaluation harness for batch-3 Cebik designs.

Runs a builder through the PyNEC reference engine and all four momwire solver
bases, printing driving-point Z and peak gain for each, and CATCHING any
exception so we can see exactly where a momwire basis chokes on a geometry/feed
(the whole point of this batch: find holes in the methodology).

Usage:
    python scratch/eval_engines.py cebik.quad_3el
    python scratch/eval_engines.py cebik.four_square --nsegs 21 41 81
"""

from __future__ import annotations

import argparse
import importlib
import traceback

from momwire import TriangularSolver, SinusoidalSolver, BSplineSolver

from antenna_designer.engines import PyNECEngine, MomwireEngine

MOMWIRE_ENGINES = [
    ("triangular", dict(solver=TriangularSolver)),
    ("sinusoidal", dict(solver=SinusoidalSolver)),
    ("bspl-d1", dict(solver=BSplineSolver, solver_kwargs={"degree": 1})),
    ("bspl-d2", dict(solver=BSplineSolver, solver_kwargs={"degree": 2})),
]


def _load_builder(dotted):
    """cebik.quad_3el -> designs.cebik.quad_3el.Builder."""
    mod = importlib.import_module(f"antenna_designer.designs.{dotted}")
    return mod.Builder


def _fmt_z(zs):
    return ", ".join(f"{z.real:7.1f}{z.imag:+7.1f}j" for z in zs)


def _run(label, make_engine):
    try:
        eng = make_engine()
        zs = eng.impedance()
        try:
            g = eng.far_field(n_theta=90, n_phi=360, del_theta=1, del_phi=1).max_gain
            gtxt = f"{g:6.2f} dBi"
        except Exception as e:  # noqa: BLE001
            gtxt = f"gain ERR: {type(e).__name__}: {e}"
        print(f"  {label:12s} Z=[{_fmt_z(zs)}]  gain={gtxt}")
    except Exception as e:  # noqa: BLE001
        print(f"  {label:12s} *** {type(e).__name__}: {e}")
        if "--tb" in __import__("sys").argv:
            traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design")
    ap.add_argument("--nsegs", type=int, nargs="*", default=[21])
    ap.add_argument("--ground", default=None)
    ap.add_argument("--tb", action="store_true")
    args = ap.parse_args()

    Builder = _load_builder(args.design)

    for n in args.nsegs:
        print(f"\n=== {args.design}  nominal_nsegs={n}  ground={args.ground} ===")

        def builder():
            b = Builder()
            b.nominal_nsegs = n
            return b

        _run("pynec", lambda: PyNECEngine(builder(), ground=args.ground))
        for name, kw in MOMWIRE_ENGINES:
            _run(name, lambda kw=kw: MomwireEngine(builder(), ground=args.ground, **kw))


if __name__ == "__main__":
    main()
