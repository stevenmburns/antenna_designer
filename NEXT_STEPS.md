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

| limitation | affected designs (today) | severity |
|---|---|---|
| pure closed loops (no degree-1 endpoint, no degree-3+ junction in any component) | `bowtie`, `freq_based.delta_loop`, `freq_based.diamond_loop`, `freq_based.inv_delta_loop`, `freq_based.folded_invvee`, `freq_based.delta_loop_slanted{,2,3}` | translator-only fix, no upstream pysim work needed |
| multiple excitations (`ex_card` voltage on more than one segment) | every array design — `invveearray`, `moxonarray`, `yagiarray`, `bowtiearray*`, `delta_looparray*`, `hentenna_array`, `hourglass_array`, `folded_invveearray`, `diamond_loop_turnstile` | deeper PysimEngine + likely upstream pysim change |
| transmission-line cards (`tl_card`) | `freq_based.delta_looparray_with_tls` | upstream pysim has no equivalent today |

## Next branch: closed-loop support in the translator

**Goal**: lift the "closed loops with no junctions/endpoints" `NotImplementedError` in `flat_wires_to_polylines` so `bowtie` and the delta/diamond loops can be solved by `PysimEngine`.

### What "closed loop" means here

A connected component in the wire graph where every node has degree 2 — no degree-1 free ends, no degree-3+ junctions. The current walker can't start because it iterates only boundary nodes. The excitation lives on one of the loop's edges.

Concrete shape, from `bowtie.build_wires()`:

```
A → B → C → D → E → A     (closed loop, all nodes degree 2)
                  ^
                  feed gap somewhere along this chain
```

(Bowtie is actually two triangles sharing a feed gap, so it has *two* loops sharing two nodes — degenerate-tee territory. Single delta loop is the canonical pure-cycle case.)

### Approach

1. **Detect pure cycles**. After the existing junction/endpoint walker finishes, any remaining unwalked edges belong to pure-cycle components. Group those edges by connected component.

2. **Choose a cut point** in each cycle. Two reasonable rules; pick whichever makes feed placement cleanest:

   - **Cut at the excited edge** if the cycle contains the feed. The excited tuple becomes its own polyline (start ↔ end of the feed edge), and a *second* polyline runs the long way around the loop, connecting the same two nodes. The two polylines share both endpoints, which makes those endpoints degree-2 junctions in pysim's `junctions=` sense. Cleanest because the feed-arclength computation is identical to the open-polyline case.

   - **Cut at an arbitrary node** if the cycle has no feed (parasitic loop coupling). Same junction-pair pattern; feed handling unchanged.

3. **Emit the junction list**. The two cut endpoints each get a 2-element junction record `[(loop_polyline_0, "start"), (loop_polyline_1, "start")]` and `[(loop_polyline_0, "end"), (loop_polyline_1, "end")]`. Pysim's KCL Lagrange multiplier rows then enforce current continuity around the loop.

4. **Compose with existing junction logic**. The translator already handles arbitrary-degree junctions for the non-cycle case (hentenna, fandipole). The cycle case is conceptually "two polylines sharing both endpoints" — the existing junction emission code should compose if we just append to the `junctions` dict already in flight.

### Bowtie wrinkle

Bowtie's two triangles share a common feed-gap edge: degree-2 nodes everywhere within each triangle, but the two feed-gap endpoints are each degree-3 (one edge of each triangle plus the feed gap). So it's already a junction-bearing component, not a pure cycle — the current translator's degree check should already accept it, and the failure mode reported in the inventory is the *no-boundary* check tripping because there are still no degree-1 nodes. Need to soften that check: "has at least one boundary node OR was reachable via junctions". Bowtie should fall out of step 1 for free once the walker considers junction nodes as valid starting points for cycle traversal too.

Verify on bowtie specifically — it may need no special handling beyond making the no-boundary error less paranoid.

### Tests

- Translator structure check on `bowtie`, `freq_based.delta_loop`, `freq_based.diamond_loop`: expected polyline / junction counts.
- Loose impedance bound for SinusoidalPySim vs PyNECEngine (free space) on `delta_loop`. Delta loops are textbook ~100 Ω at resonance; a < 15 % R / < 15 Ω X bound should be safe.
- Existing dipole/hentenna/fandipole tests continue to pass — the cycle code path doesn't touch them.

### Out of scope for that branch

- Multi-feed arrays (bucket 2 above). Separate PR, needs upstream pysim work.
- `tl_card` (bucket 3). Separate concern; likely "document as PyNEC-only" rather than implement.
- Auto-refining short feed segments. The fandipole's `n_seg1=1` feed-gap trick that breaks Triangular but works under Sinusoidal is a basis-selection question, not a geometry-translation question.

## Other follow-ups (lower priority)

- **CLI `--basis triangular|sinusoidal|bspline` flag** on the analysis subcommands. Today `--engine pysim` always uses Triangular; choosing Sinusoidal requires the Python API. Straightforward add — mirror how `--ground` is plumbed.
- **Far-field for the new designs**. Stage 2b validated the pattern math on the dipole; once tee-junction geometries are stable, re-run the directivity cross-check on hentenna and fandipole and add a corresponding test.
- **Multi-feed PysimEngine**. Two viable shapes: (a) N independent solves, superposing per-feed for the impedance matrix Y; (b) genuine multi-source solve in pysim, requires upstream changes. Worth a design sketch before either.
