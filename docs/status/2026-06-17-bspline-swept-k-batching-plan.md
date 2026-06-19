# BSplinePySim swept-k batching — diagnosis and plan

Date: 2026-06-17
Author: smburns47

## Implementation update (2026-06-19)

Implemented in the `pysim` submodule on branch `bspline-swept-k-batching`
(commits: leggauss memoization → same-edge hoist → swept-k reg batching).

**Re-measured first — and it changed the approach.** Open question #1 ("is
the per-k path accelerator-bound?") resolved to **no**: on a 41-point
trap_fan sweep the C++ off-edge kernel was only ~8% of the time and the C++
`assemble_Z_bspline` ~3%; the swept bottleneck was **Python** — recomputed
`leggauss` nodes (738 calls, ~159 ms cum) and per-k same-edge static + reg
work. So the plan's literal **(n_k, N, N) pure-Python batched assembly was
NOT implemented**: it would be 9.5 GB at bowtie N=41 *and* regress, since it
replaces the cheap per-k C++ assembly with slower numpy. Instead the
k-independent work was hoisted out of the per-k loop and only the cheap
Python reg-kernel einsum was batched over k (chunked for memory); the C++
off-edge/assembly and the LU solves stay per-k (Step 4 option 1, as the plan
recommended).

What landed, against the plan's steps:
- **Step 1 (batched J):** in spirit — k-independent J pieces (static moments,
  reg quadrature geometry) hoisted/shared; reg moments batched over k via
  `_seg_seg_reg_moments_from_geometry_swept`. The k-dependent off-edge block
  stays the per-k C++ call (cheap).
- **Step 2 (static once):** done — the plan said this was "already once per
  sweep"; it was **not** (`_build_J_blocks` recomputed it every k). Now once.
- **Step 3 (batched Z):** deliberately **not** as (n_k, N, N) — see above.
- **Step 4 (solve loop):** unchanged (already per-k).
- **Step 5 (enrichment):** `compute_impedance_swept` already supports it (it
  loops `compute_impedance`). `compute_y_matrix_swept` still raises for
  `use_singular_enrichment` — a pre-existing, opt-in, bspline-only gap that
  is **not** a Triangular-retirement blocker (Triangular has no enrichment),
  so it was left as-is.
- **Step 6 (source):** already k-independent, unchanged.
- **Memory:** the only new allocation (the `exp(-jkR)` phase intermediate) is
  k-chunked under a byte budget.

Result — per-k swept cost, mean of reps, `OPENBLAS_NUM_THREADS=1`:

```
design                segs   Bs1/Tri before*   Bs1/Tri after
trap_fan (small)      ~100   ~5x               2.06x
fandipole (medium)    ~423   ~2.6x (@300)      1.77x
bowtiearray (large)   ~696   ~1.7x             1.69x   (no regression)
```
*before = the benchmark table below / the plan's original ratios.

trap_fan 41-k swept, cumulative: **Bspline=1 353 → 123 ms (−65%)**,
Bspline=2 417 → 149 ms (−64%); impedance unchanged. Verified roundoff-equal
to the per-k path (~7e-13 relative); 104 pysim + 1058 antenna_designer tests
pass, plus a new `test_bspline_fandipole_swept_matches_per_freq`.

Remaining lever (not done): on *tiny* designs (<~50 segs) Bs1 is still ~7×
Tri, but only ~1.4 ms/k absolute — the bottleneck there is the per-k C++
off-edge call + solve, which Triangular batches over k via a C++ kernel that
accepts a k-array. Matching that needs a batched C++ off-edge kernel for the
bspline basis (the "heavy lift" the plan flagged); deferred as low-value
(sub-2 ms/k).

## TL;DR

The process-level caches added in pysim PRs #80 (geometry) and #81
(basis polynomials + inner-loop vectorize) gave a real one-shot
speedup on single `impedance()` calls, but on **41-point swept-k
sweeps** — the dominant interactive UI path — BSplinePySim is still
5–6× slower than `TriangularPySim` on **small** designs (fan-dipole
class, ≤300 segments). On **large** designs (bowtiearray2x4, ~1400
segments) the gap shrinks to ~1.5–1.7× because the actual per-k
numerical work dominates the fixed Python overhead. The caches help
everything that is k-independent (geometry build, basis polynomial
extraction), but the **Z-matrix assembly** runs in a Python loop over k
inside `compute_impedance_swept`. `TriangularPySim` has a fully batched
swept-k path (`_build_J_blocks_batch` → `_assemble_Z_batch` →
`_solve_with_kcl_batch`) that BSplinePySim does not. **Porting the
batched path mostly helps small designs**, and is also a prerequisite
for retiring `TriangularPySim` in favor of d=1 BSpline.

## Benchmark (run 2026-06-17)

41-point frequency sweep around design center, 3 reps after warm,
mean ± stdev. `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=<phys cores>`,
matching the UI/server env.

```
case                             |        Tri         |        Bs1         |        Bs2         |        Sin
----------------------------------------------------------------------------------------------------------------------
trap_fan N=21 (84 segs)          |      64 ms (±   7) |     339 ms (±  12) |     417 ms (±  26) |      93 ms (±   3)
trap_fan N=81 (~300 segs)        |     666 ms (±  13) |    1761 ms (±  24) |    2522 ms (± 110) |     854 ms (±   4)
bowtiearray2x4 N=21 (1392 segs)  |   15708 ms (±  31) |   22702 ms (± 784) |   26987 ms (±  72) |   17778 ms (± 192)
```

Bowtiearray2x4 N=41 (2704 segs), **11-pt sweep** (k=11 instead of 41,
to keep wall-clock and memory tractable):

```
case                                       |        Tri          |        Bs1          |        Bs2          |        Sin
------------------------------------------------------------------------------------------------------------------------------------
bowtiearray2x4 N=41 (2704 segs, k=11)      |  21460 ms (± 167)   |  28189 ms (± 527)   |  32326 ms (± 114)   |  30987 ms (± 328)
```

Per-k cost (dividing by the sweep length) is the apples-to-apples
comparison:

```
case                                |   Tri ms/k | Bs1 ms/k | Bs2 ms/k | Sin ms/k
-----------------------------------------------------------------------------------
trap_fan N=21  (84 segs, k=41)      |        1.6 |      8.3 |     10.2 |      2.3
trap_fan N=81  (~300 segs, k=41)    |       16.2 |     43.0 |     61.5 |     20.8
bowtie N=21   (1392 segs, k=41)     |      383   |    554   |    658   |    434
bowtie N=41   (2704 segs, k=11)     |     1951   |   2563   |   2939   |   2817
```

Scaling factor N=21 → N=81 trap_fan (4× in N, ≈64× in solve cost):

| Engine | Ratio | What it means |
|---|---|---|
| Tri | 10.4× | Batched path: fixed sweep overhead is small, scaling is mostly Z assembly + solve |
| Sin | 9.2× | Batched (sinusoidal also assembles all k's at once) |
| Bs1 | 5.2× | Big per-k Python overhead — amortizes as N grows |
| Bs2 | 6.0× | Same Python overhead, slightly worse scaling from `(d+1)² = 9` blocks |

Ratio of `<engine>` / Tri at each design size (per-k):

| Design | Bs1 / Tri | Bs2 / Tri | Sin / Tri |
|---|---|---|---|
| trap_fan N=21 (84 segs) | 5.3× | 6.5× | 1.5× |
| trap_fan N=81 (~300 segs) | 2.6× | 3.8× | 1.3× |
| bowtiearray2x4 N=21 (1392 segs) | 1.45× | 1.72× | 1.13× |
| bowtiearray2x4 N=41 (2704 segs) | 1.31× | 1.51× | 1.44× |

**Two regimes, two stories.** On *small* designs (~few hundred segments
or fewer) the Bs1/Bs2 path is dominated by fixed per-k Python overhead
and runs 3–7× slower than Tri's batched path. On *large* designs
(>1000 segments) the per-k cost is dominated by the actual O(N²)
Z-matrix assembly work and the gap closes to 1.3–1.7×. The batching
port would mostly buy us "small designs converge fast in the UI" —
bowtie-sized designs are *already* roughly competitive without it.

**Sinusoidal scaling anomaly**: Sin's per-k cost grows the fastest of
all four engines from bowtie N=21 → N=41 (434 → 2817 ms/k, 6.5× for
2× N), surpassing Bs1 and approaching Bs2. The other engines scale
~5×. Likely the closed-form sinusoidal charge integrals have an
O(N³) component (one segment's basis interacts with all others on the
same wire), where the polynomial-basis path is more localized. Worth
profiling if Sin becomes interesting for bigger problems — but not
on the critical path of the BSpline batching work.

## What the caches do and don't help

Stages of one `compute_impedance_swept(k_array)` call in `bspline.py`:

| Stage | k-dependent? | Status after PRs #80 / #81 |
|---|---|---|
| `_build_geometry` | no | **cached** (process-level `_GEOMETRY_CACHE`) |
| `_build_basis_polynomials` | no | **cached** (process-level `_BASIS_POLY_CACHE`, inner loop vectorized) |
| Static moments (closed-form) | no | already 1× per sweep |
| `J_{pq}` blocks (full kernel) | **yes** — `exp(-jkR)` | per-k Python loop |
| `Z_A` and `Z_Φ` assembly | **yes** | per-k Python loop |
| Singular enrichment block | **yes** | per-k Python loop (also routed through C++ `assemble_Z_enrich`) |
| KCL Schur-complement solve | **yes** | per-k Python loop, 2× `scipy.linalg.solve` per k |

The Python-loop wrapper around the per-k stages is in
`compute_impedance_swept`. From the prior status doc and PR #81 commit
notes: it's structurally a list-comp over `k_array` with full geometry
and basis tuple passed in to each iteration.

The k-independent caches save *only* the work in rows 1–3. For a
41-point sweep at N=84, that work is ≈30–50 ms total (one-time), and
the per-k Python overhead is ~7–10 ms × 41 ≈ 300–400 ms — which
matches the observed Bs1 number.

## Why `TriangularPySim` is fast on small problems

`triangular.py:_build_J_blocks_batch(geom, k_array)` evaluates all
`J_{pq}` integrals across all k's in one shot via numpy broadcasting:

- Gauss-Legendre quadrature node positions and R values are
  geometry-only — computed once, shape `(n_pairs, n_qp_pair²)`.
- `exp(-jkR)` is the only k-dependent factor; broadcast to
  `(n_k, n_pairs, n_qp_pair²)`.
- Reduction over quadrature axis happens in one vectorized call;
  result `(n_k, n_segs, n_segs)` for each `(p, q)` block.

`_assemble_Z_batch` then builds the full `(n_k, N, N)` Z stack via
broadcasting. `_solve_with_kcl_batch` factors and solves all 41 systems
— but the LU factorization itself is still done in a Python loop over
the k axis (scipy doesn't batch LU), so the *solve* isn't actually
batched, just the assembly. The dominant savings come from assembly.

## Porting plan for `BSplinePySim`

Scope is bigger than triangular because bspline has more moving parts.
Order them by independence so each step is verifiable on its own.

### Step 1: Batched `J_{pq}` block evaluator

Mirror `_build_J_blocks_batch` from triangular. Generalize to
`p, q ∈ {0, ..., d}` (4 blocks for d=1, 9 for d=2). Same broadcasting
strategy: geometry-only R + Gauss-Legendre nodes computed once, phase
factor `exp(-jkR)` broadcast over the k axis.

**Verifiable by**: at a single k, the batched evaluator should match
the per-k evaluator's output to roundoff.

### Step 2: Batched static moments

Static moments are already k-independent (they're the analytic
closed-form pieces for same-edge pairs). No change needed — they're
already computed once per sweep.

### Step 3: Batched `Z_A` and `Z_Φ` assembly

`Z_A` is the vector-potential block (mass matrix scaled by `jωμ`),
`Z_Φ` is the scalar-potential block. Both are built from the J_{pq}
blocks via fixed combinatorial sums over polynomial coefficients C[m,
a, p]. With J_{pq} now (n_k, N, N), Z_A and Z_Φ become (n_k, N, N) via
the same tensor contractions but with one extra (broadcast) k axis.

**Verifiable by**: at a single k, batched Z matches per-k Z to roundoff.

### Step 4: Batched solve

Two scipy.linalg.solve calls per k (Schur complement of KCL constraint
+ final back-substitution). Three options:

1. **Python-loop the solves**: keep batched assembly, but loop the LU
   factor + solve per k. This is what triangular does and it's fine for
   41 k's — the matrix is already in memory.
2. **`np.linalg.solve` with broadcast**: accepts (n_k, N, N) stack and
   does the factorize + solve in numpy. Whether this is faster than
   the Python loop depends on numpy's internal batching strategy
   (LAPACK doesn't batch).
3. **Reuse LU factorization**: the Schur complement matrix is
   k-dependent (it's a function of Z(k)), so LU reuse across k doesn't
   apply. Same conclusion the LU-reuse investigation reached on PR #81.

Recommend (1) — matches Triangular, no new surface area.

### Step 5: Singular enrichment batching

The XFEM enrichment path adds extra rows/cols to Z for K≥3 junctions.
Currently routed through `_accelerators.assemble_Z_enrich` (C++). For
batched swept-k, either:

- Extend the C++ accelerator to accept a k-array. Probably the right
  long-term move, but a heavy lift.
- Python-loop the enrichment block while batching the polynomial-basis
  blocks. The enrichment block is small (one row+col per qualifying
  junction), so this is cheap.

Recommend the second initially — get to a working batched path, profile,
extend the accelerator if it becomes the bottleneck.

### Step 6: Smoothed-cos² feed source

Source vector `v_m = Φ_m(s_f)` or integral of cos²-windowed bump
against basis. **k-independent.** Compute once, reuse for all k's.

Already true on the current Python-loop path (it's lifted out of the
inner loop). No change.

## Risk and effort

Estimated 1–2 days of focused work. Risks:

- **Memory at large N**: at bowtie N=41 (2704 segs), batched Z is
  `(41, 2704, 2704)` = 9.5 GB complex128. Need to chunk the k axis
  (process in groups of e.g. 8 k's) or stay with the per-k loop for
  the assembly stage of huge problems and only batch J_{pq}.
  Triangular handles this with a `chunk_size` parameter.
- **Numerical equivalence**: every existing convergence test should
  produce bit-for-bit (or single-ulp) identical results before and
  after the port. The cache PRs were validated this way; same
  approach here.
- **XFEM regression**: K≥3 enrichment paths have variants ("raw",
  "stable", "tikhonov", "auto") — each needs to keep working in the
  batched path. The "auto" variant runs *two* solves (probe + final)
  — both should batch.

## Why this beats the alternatives

- **Removing TriangularPySim** (in favor of d=1 BSpline): can't happen
  until bspline has the batched-swept-k path, since the UI's post-dwell
  41-point sweep is what makes Triangular load-bearing.
- **Just deleting the Python-loop sweep**: not an option — it's the
  whole point of the UI's interactive frequency-sweep workflow.
- **Cython / numba for the inner loop**: would help a bit, but the
  vectorize PR (#81) already proved that the inner basis-build loop
  responds well to numpy broadcasting. The right primitive is numpy
  vectorization on the k axis, not low-level loop acceleration.

## Followups

- d=0 extension — **investigated and shelved (2026-06-17).** Not gated on
  this work, and not happening: the original plan's premise (drop the
  scalar potential at d=0) gives a degenerate solver, and the correct
  pulse-basis charge term needs a staggered dual-mesh assembly that is
  essentially a standalone pulse solver's charge core — not worth it for
  the least-useful (O(1/N)) degree. See `pulse_basis_d0_nodal_charge.md`
  in pysim for the full diagnosis. The convergence-curve points are
  d=1/d=2 (and possibly d=3).
- Triangular retirement — replace with d=1 BSpline once feature
  parity (batched sweep + smoothed source + singular enrichment) is
  reached. Reduces solver-engine surface area.
- Higher-degree B-splines (d=3, d=4) — easier to add once the batched
  path exists since `_build_J_blocks_batch` is degree-parametric.

## Open questions

1. Does the existing C++ `assemble_Z_bspline` accelerator help in the
   per-k path enough that the batched-Python path is a regression?
   Need to measure — if yes, the port must integrate batching into the
   accelerator rather than going back to pure Python.
2. Memory pressure on bowtie N=41 will determine whether chunking is
   mandatory or a tuning knob. Bowtie N=21 (1392 segs, k=41) ran cleanly
   at ~16–27 seconds per sweep across all four engines. Bowtie N=41
   (2704 segs, k=11) ran in ~21–32 seconds. A full N=41 with k=41
   batched Z would be ~9.5 GB complex128 — chunking the k axis is
   **mandatory** for that size.

   Note: the web server already chunks sweeps, but for *time*, not
   *memory*. `web/server.py` splits the pysim `/sweep` freq list into
   groups targeting `_CHUNK_TARGET_MS = 500` ms each (server.py:120),
   starting from an 8-chunk heuristic (`chunk_size = max(1,
   len(freqs) // 8)`, server.py:522) and re-tuning from observed
   per-freq cost after each chunk (server.py:556–558). This is for
   cancellation granularity and threadpool/memory pacing under rapid
   slider drags — each chunk still calls `compute_impedance_swept`
   once with all its k's. As a side effect it roughly bounds k per
   solver call (slow/large designs naturally get small chunks), so it
   *partially* caps the `(n_k, N, N)` allocation — but it's tuned for
   wall-time, not bytes, so the batched path still needs its own
   memory-aware `chunk_size` knob inside `compute_impedance_swept`
   (mirroring triangular's) rather than relying on the server layer.
3. Is there a non-trivial geometry where the singular enrichment cost
   per k dominates the polynomial-basis cost? If so, the C++
   accelerator extension is more urgent than this doc suggests.

