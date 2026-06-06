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

## Next branches (rough priority order)

### 1. Cross-engine pattern comparison in the CLI

Today `compare --engine pysim --builders dipole invvee` runs both builders through *one* engine. The Python API already lets `compare_patterns` take a heterogeneous list of engine instances; the gap is just CLI plumbing. Extend the `compare` subcommand to accept `--engines pynec pysim` and take the cross-product with `--builders` (labels become `dipole/pynec`, `dipole/pysim`, …). Same change naturally subsumes the previously-listed `--basis triangular|sinusoidal|bspline` follow-up: spell engine choices as `pynec`, `pysim:triangular`, `pysim:sinusoidal`, `pysim:bspline` so the basis is part of the engine identity rather than a separate orthogonal flag. Small, no upstream changes, immediate cross-validation value.

### 2. Named parameter variants per builder

Designs currently export one default param dict per module. Real usage (different bands, different element counts, swept-then-frozen configurations) wants multiple named variants checked into the same file. Proposal:

- A design module can expose a `VARIANTS: dict[str, dict]` mapping name → param dict alongside the existing default. The default stays the unnamed variant.
- Builder selector syntax on the CLI becomes `builder[:variant]`, e.g. `dipole:80m`, `hentenna:wide`. No colon → default variant (back-compatible).
- The builder registry resolves `name:variant` to `(builder_callable, params_override)` and threads the override through the existing builder API.

Open question: should variants compose with per-flag overrides (`--set length=5.2`)? The two solve different problems — named variants are for reproducible saved configurations, ad-hoc overrides are for sweeping. Probably both, but named variants first.

### 3. Multi-feed PysimEngine (and where the interface code should live)

This is the dominant blocker — 16 of the currently-rejected designs are multi-feed arrays. Two viable shapes, unchanged from before:

- **(a) N independent solves, superpose for Y.** Drive each feed in turn with the others open/shorted (the choice matters for what Y means), recover the multi-port impedance matrix column-by-column. Works entirely above pysim's existing single-source API. Cheapest path; the answer is exact for linear MoM.
- **(b) Genuine multi-source solve in pysim.** Requires upstream changes — basis assembly and the RHS construction need to know about multiple excitations simultaneously. More natural and avoids N solves, but couples our roadmap to a pysim release.

Worth a design sketch before either. Recommend prototyping (a) first because it's reversible and validates the multi-port plumbing in `antenna_designer` independently of upstream churn.

**Where the interface code lives.** Once multi-feed works, a real question opens up: the geometry translator (`flat_wires_to_polylines`, the closed-loop cut, the multi-port Y assembly) is antenna-agnostic and would benefit any pysim user, not just us. Tempting to move it upstream. **Don't do this yet** — moving code across repos creates an API contract that's expensive to iterate against, and we're still discovering the right shape (multi-feed will almost certainly reshape the translator's interface). Revisit after multi-feed lands and the translator's signature has been stable for a release or two.

### 4. Far-field for the new designs

Stage 2b validated the pattern math on the dipole; once tee-junction geometries are stable, re-run the directivity cross-check on hentenna and fandipole and add a corresponding test. Lower priority — falls out naturally as a side effect of #1 once the CLI can compare engines on these designs.
