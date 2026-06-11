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
| ~~multiple excitations (`ex_card` voltage on more than one segment)~~ | ~~every array design~~ | **done** — translator emits a `feeds` list, `PysimEngine.impedance()` returns per-port Z |
| ~~transmission-line cards (`tl_card`)~~ | ~~`freq_based.delta_looparray_with_tls`~~ | **done** — `impedance()` only; sweep raises `NotImplementedError` |

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

## ~~Next branch: cross-engine pattern comparison in the CLI~~ — landed

The `compare_patterns` subcommand accepts `--engines` plural, with each spec spelled as `pynec`, `pysim`, or `pysim:triangular|sinusoidal|bspline` (basis is part of engine identity, not a separate flag). Builders × engines combine via **numpy-style broadcasting** in `broadcast_pairs` (cli.py), not literal Cartesian product: equal lengths zip pairwise, length-1 broadcasts against the other, other mismatches reject. Labels are `bname/espec` when both vary, otherwise whichever varies. Covered by `tests/test_engine_spec.py` (23 tests).

Surprise relative to the plan: broadcasting beats cross-product because it lets you express specific pairings (`--builders A B C --engines E1 E2 E3` zips into 3 chosen pairs); a true Cartesian product would always yield 9. If we ever want both, add an opt-in `--cross-product` flag on top.

## ~~Next branch: named parameter variants per builder~~ — landed

The convention turned out simpler than the original `VARIANTS: dict[str, dict]` proposal: a Builder class exposes variants as class attributes whose name ends in `_params` and whose value is a `Mapping` (typically `MappingProxyType`). The unnamed default is `default_params`; named variants are `opt_params`, `s07_params`, `current_physical_params`, etc. CLI selector syntax is `builder[:variant]` (cli.py:90 `get_builder`); `:default` and no colon both pick `default_params`. `list_variants` discovers the available names for error messages.

In practice this nests with the existing builder-resolution rules (local/library × explicit/implicit `Builder`), so `freq_based.invvee:dipole` works the same way `freq_based.invvee` does.

Open question deferred: compositing variants with per-flag overrides (`--set length=5.2`). Not implemented; ad-hoc sweeping still goes through `sweep`/`optimize`.

## ~~Next branch: multi-feed PysimEngine~~ — landed

Discovered to already be working when re-auditing the design tree: 34/35 designs solve cleanly through `PysimEngine.impedance()`, including every multi-feed array previously listed as blocked (`invveearray`, `moxonarray`, `yagiarray`, `bowtiearray{,1x2,2x4}`, `delta_looparray{,_1x4,_1x4_grouped,_2x2}`, `hentenna_array`, `hourglass_array`, `folded_invveearray`, `diamond_loop_turnstile`). The geometry translator emits a `feeds: list[(polyline_idx, arclength, voltage)]` and `PysimEngine.impedance()` returns a list of per-port impedances; `impedance_sweep` normalises its return to `(n_k, n_feeds)` to match PyNECEngine.

The only remaining limitation is `tl_card` (1 design: `delta_looparray_with_tls`).

Cross-validation at design freq, free space (R, X in Ω) — sampled multi-feed arrays vs PyNEC:

| design  | feeds | PyNEC range          | pysim Triangular range | max ΔR | max ΔX |
|---------|-------|----------------------|------------------------|--------|--------|
| invveearray | 4 | 48.0…55.4, −7.8…−3.0 | 47.1…54.8, −9.5…−4.8   | 0.95   | 1.82   |
| moxonarray  | 4 | 43.8…44.3, −28.5…−24.9 | 39.9…40.4, −30.0…−26.4 | 3.95   | 1.51   |
| yagiarray   | 4 | 81.6…81.9, +2.1     | 83.8…84.1, +0.7…+0.8  | 2.23   | 1.34   |

Same ballpark agreement as the closed-loop work (~few % R, a few Ω X). Cross-validation tests live in `tests/test_pysim_engine.py`.

**Where the interface code lives.** Same call as before: don't move the translator upstream yet. The shape just settled; let it bake.

## ~~Next branch: `tl_card` support in PysimEngine~~ — landed

`PysimEngine.impedance()` now handles transmission-line cards by extracting the multi-port Y matrix via **N independent solves** (one per port, V=1 at driven port, V=0 elsewhere), stamping each TL's 2×2 admittance contribution at its endpoint pair, then reducing back to the driven-port impedance via nodal analysis with passive-port currents constrained to zero. The N-solves approach was forced by an upstream limitation: pysim's `compute_y_matrix` doesn't yet support junctions, and every TL design has junctions (the delta loops).

The path also fixed a latent `AttributeError` — `PysimEngine.__init__` used to call `builder.build_tls()` before `build_wires()` had run, blowing up on builders that populate `self.tls` inside `build_wires`. Order is reversed now.

**Cross-validation surprise.** PyNEC and pysim disagree wildly on `delta_looparray_with_tls` at default params (PyNEC: −77 −18255j; pysim: +55 −3j). Both engines stay self-consistent across frequency and TL length sweeps — this is a genuine modeling-convention difference, not numerical noise. The most likely root cause: NEC2's `tl_card` treats TL endpoints as **segment-level** ports (network attached to the segment as a whole) while my pysim post-processing treats them as **basis-level** ports (delta-gap at the wire midpoint). On simple geometries the two coincide; on this design the central driver is a 10cm gap with effectively zero coupling to the loops in the antenna's own Y matrix, so the TL transformation dominates and the segment-vs-basis distinction blows up the result. Reproducer: `Y[loop, driver] ≈ 3e-7` (essentially decoupled) — every Ω of driver impedance comes from how you stamp the TLs, so the conventions diverge maximally.

What this means going forward: the multi-port Y reduction is mathematically clean (verified: Y symmetric, passive-port `I_ext=0` constraint exact, `coeffs[m] = 1/Z` at the feed for single-port). Self-consistency tests are in `tests/test_pysim_engine.py`; strict PyNEC numerical agreement isn't currently achievable on the one available test fixture. If we ever land another TL design where loop-driver coupling is non-negligible, retry the comparison.

`impedance_sweep` with TLs raises `NotImplementedError` — `compute_y_matrix_swept` exists upstream, but per-k stamping plus reduction is its own piece of code that no current design needs.

## Next branches (rough priority order)

### 1. Strict PyNEC cross-validation for `tl_card`

`PysimEngine.impedance()` runs cleanly on `delta_looparray_with_tls` but the numerical answer doesn't match PyNEC (segment-vs-basis port convention — see the closed branch above). Two follow-ups, both optional:

- Add a TL design where the in-antenna coupling between TL endpoints isn't near-zero, then re-run the comparison; agreement should be much tighter when the antenna's own Y has meaningful off-diagonal terms.
- Implement segment-averaging at the TL endpoints (averaged current over the TL-end segment instead of basis coefficient at the midpoint). Closer to NEC2's convention; may reconcile.

### 2. `impedance_sweep` with TLs

`compute_y_matrix_swept` exists upstream — wire it up + per-k TL stamping + reduction. No current design needs this; add when one does.

### 3. Junction support in pysim's `compute_y_matrix`

The N-independent-solves path costs N× the LU factor cost. pysim's batched `compute_y_matrix` would do it in one factor + N back-substitutions, but currently rejects junctions. The Schur-complement KCL solve in `_solve_with_kcl` needs a matrix-RHS generalisation. Upstream change; only worth it if larger N-port problems appear.

### 4. Far-field for the new designs

Stage 2b validated the pattern math on the dipole; once tee-junction geometries are stable, re-run the directivity cross-check on hentenna and fandipole and add a corresponding test. Lower priority — now that cross-engine `compare_patterns` is in the CLI, this is largely a "run it and write a test" task.
