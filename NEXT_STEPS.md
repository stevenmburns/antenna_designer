# Next steps — antenna_designer

Living roadmap of what's done and what's left on the multi-engine refactor. Updated as work lands.

## Where we are

`antenna_designer` exposes two simulation backends behind a common `SimulationEngine` interface:

- **`PyNECEngine`** — the original NEC2/PyNEC path. Default backend. Supports free space / PEC / finite-Sommerfeld ground via the `ground=` parameter, multi-element arrays, transmission-line cards (`tl_card`), and arbitrary segment counts. This is the production backend; nothing here is restricted relative to the historical behaviour.
- **`PysimEngine`** — the pure-Python MoM solvers from the vendored `pysim/` submodule, accessed through a flat-wire-to-polyline geometry translator (`antenna_designer/geometry.py`). Selectable solver (`TriangularPySim` default, `SinusoidalPySim`, `BSplinePySim`) via the engine's `solver=` kwarg.

Selection is uniform across the Python API and the CLI: `compare_patterns([engine_instance, ...])` for ad-hoc comparisons; `engine=` factory kwarg on `sweep` / `sweep_freq` / `sweep_gain` / `sweep_patterns` / `optimize`; `--engine pynec|pysim --ground free|pec|finite|finite:eps,sigma` on every analysis subcommand.

Cross-validation at design freq, free space, against PyNEC: ≤ 0.1 dBi peak directivity agreement on the dipole; ≤ 0.005 dBi on PEC ground. Sinusoidal lands within ~10 % R and ~10 Ω X of PyNEC on the hentenna's tee-junction geometry.

## Known geometry limitations

These are the topologies `PysimEngine` currently rejects. PyNECEngine is unaffected — these only matter if you're cross-validating with pysim or using it as the production backend.

| limitation | affected designs (today) | status |
|---|---|---|
| ~~pure closed loops~~ | ~~bowtie, delta_loop, diamond_loop, inv_delta_loop, folded_invvee, delta_loop_slanted{,2,3}~~ | **done** — cut-at-feed-edge in `flat_wires_to_polylines` |
| multiple excitations (`ex_card` voltage on more than one segment) | every array design — `invveearray`, `moxonarray`, `yagiarray`, `bowtiearray*`, `delta_looparray*`, `hentenna_array`, `hourglass_array`, `folded_invveearray`, `diamond_loop_turnstile` | deeper PysimEngine + likely upstream pysim change |
| transmission-line cards (`tl_card`) | `freq_based.delta_looparray_with_tls` | upstream pysim has no equivalent today |

## ~~Next branch: closed-loop support in the translator~~ — landed

Implementation: after the existing junction/endpoint walker finishes, any unwalked edges belong to pure-cycle components. The translator floods each cycle, locates its excited edge, and cuts there — emitting the excited edge as a single-edge polyline and the rest of the loop as a second polyline running the long way back. Both cut endpoints become 2-entry junctions so pysim's KCL closes the loop.

Surprise relative to the plan: **bowtie is a pure cycle, not a degree-3 case**. The two triangles share corners at `(±y, 0)` but each triangle only contributes one edge incident at the shared corner, leaving those corners degree-2. The whole bowtie is a single 10-edge cycle with one excited segment (the lower triangle's centre gap) and one passive segment (the upper triangle's centre gap, no `ex_card`). It falls out of the same cut-at-excited-edge path with no special handling.

Cross-validation at design freq, free space (R, X in Ω):

| design | PyNEC | pysim Sinusoidal | pysim Triangular |
|---|---|---|---|
| bowtie | 188.6, +16.3 | 186.1, +14.2 | 187.7, +13.9 |
| delta_loop | 113.4, +46.1 | 110.5, +43.9 | 110.5, +42.7 |
| diamond_loop | 219.7, +60.1 | 216.8, +58.0 | 220.7, +57.6 |

Both pysim bases agree with PyNEC to ~1–3 % R and a few Ω X on the closed-loop designs (tighter than the hentenna because there are no degree-3 junctions adding extra basis-family bias). All six previously-blocked closed-loop designs now translate cleanly; the remaining 16 blocked designs are *all* multi-feed arrays (no more closed-loop blockers in the codebase).

Parasitic-only loops (a cycle with no excited segment, no other component) raise a clear `NotImplementedError` rather than crashing inside pysim. Loops with multiple excitations also raise specifically. Neither case appears in `designs/`.

## Other follow-ups (lower priority)

- **CLI `--basis triangular|sinusoidal|bspline` flag** on the analysis subcommands. Today `--engine pysim` always uses Triangular; choosing Sinusoidal requires the Python API. Straightforward add — mirror how `--ground` is plumbed.
- **Far-field for the new designs**. Stage 2b validated the pattern math on the dipole; once tee-junction geometries are stable, re-run the directivity cross-check on hentenna and fandipole and add a corresponding test.
- **Multi-feed PysimEngine**. Two viable shapes: (a) N independent solves, superposing per-feed for the impedance matrix Y; (b) genuine multi-source solve in pysim, requires upstream changes. Worth a design sketch before either.
