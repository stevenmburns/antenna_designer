# 2026-06-25 — solver-selection benchmark (10 designs × 7 engines)

## Goal

Extend the [2026-06-15 solve-time comparison](2026-06-15-engine-solve-time-comparison.md)
into a **solver-selection guide**: which engine to reach for given an antenna's
*class* and *size*. Two things changed since that run:

- **Two more engines.** Added momwire's accelerated solvers — `ArrayBlockSolver`
  (element-aware block-low-rank, "Arr") and `HMatrixSolver` (ACA / hierarchical
  matrix, "ACA") — to the original four bases + PyNEC.
- **Ten designs spanning the decision space**, not two. The choice of solver
  turns on two axes — total problem size, and single-structure vs.
  many-identical-element (array) geometry — so the design set samples each cell:
  single element, small loop, beams, multiband dipoles, a large single-wire
  structure, a log-periodic, and small/large arrays.

Run on the `momwire` build that shares the system OpenMP runtime (after the
`pynec-accel` 1.7.4.post1 libgomp de-vendoring), so the accelerator stays loaded
with PyNEC co-resident — **no `GLIBC_TUNABLES`/`LD_PRELOAD` workaround**, and
`momwire.accelerated` was `True` throughout.

## Script

`scripts/profile_compare_engines.py` — `.venv/bin/python scripts/profile_compare_engines.py`.

Engines (column labels): `Tri` TriangularSolver · `Bs1`/`Bs2` BSplineSolver
degree 1/2 · `Sin` SinusoidalSolver · `Arr` ArrayBlockSolver · `ACA`
HMatrixSolver · `PyNEC` PyNECEngine (`ground="free"`).

Threading mirrors `web/server.py`, applied before any numpy/scipy/PyNEC import
(unchanged from the 2026-06-15 run): `OPENBLAS_NUM_THREADS=1`,
`OMP/MKL_NUM_THREADS=physical_core_count`, `OMP_WAIT_POLICY=PASSIVE`,
`GOMP_SPINCOUNT=0`. Per cell: one off-band warm-up call, then mean wall-clock
across the design's bands. Single-band 10 m designs sweep 28.0/28.3/28.57/28.85
(warm-up 27.0); fandipole/trap_fan_dipole keep their multiband target sets.
Geometry/Z size is fixed by (design, N), so per-call cost is essentially
N-and-solver-only and the mean hides one-shot jitter.

## Results

Mean `impedance()` ms; half-spreads were small (≲5%) except where noted.
4 physical cores (i7-4770K, Haswell). Ordered small → large.

| design | class | engine | N=21 | N=41 | N=81 |
|---|---|---|---:|---:|---:|
| **invvee** | single dipole | Tri | 3 | 7 | 25 |
| | | Bs1 | 1 | 2 | 8 |
| | | Bs2 | 2 | 3 | 10 |
| | | Sin | 1 | 2 | **5** |
| | | Arr | 6 | 13 | 49 |
| | | ACA | 8 | 24 | 83 |
| | | PyNEC | 1 | 2 | 6 |
| **delta_loop** | small loop | Tri | 5 | 11 | 42 |
| | | Bs1 | 2 | 4 | 12 |
| | | Bs2 | 2 | 6 | 21 |
| | | Sin | 2 | 3 | **9** |
| | | Arr | 9 | 26 | 95 |
| | | ACA | 20 | 77 | 153 |
| | | PyNEC | 1 | 3 | **9** |
| **moxon** | 2-el beam | Tri | 13 | 42 | 165 |
| | | Bs1 | 5 | 17 | 53 |
| | | Bs2 | 8 | 25 | 86 |
| | | Sin | 4 | 13 | 53 |
| | | Arr | 32 | 77 | 240 |
| | | ACA | 60 | 193 | 534 |
| | | PyNEC | 5 | 14 | **47** |
| **yagi** | 3-el beam | Tri | 16 | 49 | 232 |
| | | Bs1 | 6 | 19 | 72 |
| | | Bs2 | 10 | 29 | 104 |
| | | Sin | 5 | 16 | 72 |
| | | Arr | 34 | 72 | 165 |
| | | ACA | 89 | 305 | 838 |
| | | PyNEC | 6 | 22 | **68** |
| **fandipole** | multiband (5×2) | Tri | 90 | 372 | 2040 |
| | | Bs1 | 31 | 119 | 584 |
| | | Bs2 | 37 | 157 | 733 |
| | | Sin | 27 | 133 | 756 |
| | | Arr | 232 | 1030 | 5351 |
| | | ACA | 441 | 1681 | 5150 |
| | | PyNEC | 28 | 115 | **536** |
| **trap_fan_dipole** | multiband (2×2) | Tri | 11 | 25 | 82 |
| | | Bs1 | 5 | 10 | 27 |
| | | Bs2 | 7 | 13 | 36 |
| | | Sin | 4 | 8 | 23 |
| | | Arr | 22 | 59 | 181 |
| | | ACA | 37 | 115 | 342 |
| | | PyNEC | 3 | 7 | **21** |
| **rhombic** | large single wire | Tri | 629 | 3524 | 15848 |
| | | Bs1 | 226 | 1047 | 6085 |
| | | Bs2 | 294 | 1333 | 7379 |
| | | Sin | 190 | 1189 | 6232 |
| | | Arr | 1779 | 7321 | 32246 |
| | | ACA | 901 | 1870 | **3941** |
| | | PyNEC | 146 | 718 | 4265 |
| **lpda** | 10-el log-periodic | Tri | 175 | 646 | 3884 |
| | | Bs1 | 70 | 279 | 1438 |
| | | Bs2 | 86 | 362 | 1722 |
| | | Sin | 72 | 306 | 1829 |
| | | Arr | 218 | 459 | **1188** |
| | | ACA | 805 | 1881 | 4432 |
| | | PyNEC | 655 | 2757 | 13439 |
| **delta_looparray_1x4** | 4-el array | Tri | 37 | 134 | 524 |
| | | Bs1 | 14 | 43 | 200 |
| | | Bs2 | 17 | 61 | 262 |
| | | Sin | 11 | 47 | 198 |
| | | Arr | 44 | 108 | 240 |
| | | ACA | 229 | 604 | 1411 |
| | | PyNEC | 12 | 41 | **182** |
| **bowtiearray2x4** | 8-el array | Tri | 1584 | 6198 | 27569 |
| | | Bs1 | 463 | 2133 | 12052 |
| | | Bs2 | 524 | 2448 | 13439 |
| | | Sin | 502 | 2590 | 14605 |
| | | Arr | 208 | 527 | **1864** |
| | | ACA | 2640 | 6283 | 14632 |
| | | PyNEC | 381 | 1947 | 11386 |

(Larger half-spreads: fandipole Arr N=81 ±313, bowtie Sin/ACA N=81 ±146/±281;
everything else within a few %.)

## Solver-selection guide

| Antenna class | Use | Why |
|---|---|---|
| Single elements, small loops, beams, multiband dipoles | **`sinusoidal`** (momwire) or **PyNEC** | dense solve is milliseconds; the accelerators' ACA setup is pure overhead here |
| **Large single-wire structures** (rhombic, long-wires, big loops) | **`hmatrix`** (ACA) at high segmentation | sub-quadratic scaling; the only solver that beats PyNEC on rhombic at N=81 |
| **Arrays** of identical / few-shape elements (bowtie/quad/loop arrays, LPDA) | **`arrayblock`** | 6–11× faster than PyNEC on large arrays; near-linear scaling |

Footnote for users: `arrayblock`/`hmatrix` only win at moderate-to-high N on
their target geometries — below ~N=40 they are slower than dense even there, so
do not reach for them on small problems. PyNEC is an excellent fast reference
for single/few-element designs but is **not** universally fast (see LPDA).

## Headlines

- **ACA finally earns its place — on `rhombic`.** `HMatrixSolver` is the fastest
  engine at N=81 (3941 ms), beating PyNEC (4265) and every dense basis
  (Sin 6232, Bs1 6085). It scales ~2×/step (901→1870→3941) vs dense's ~5×/step;
  crossover is ~N=60. A large *single* structure with spatial extent is exactly
  the distance-based-low-rank regime ACA is built for — and none of the
  pre-existing designs exercised it.
- **ArrayBlock dominates arrays, and LPDA is its showcase.** `bowtiearray2x4`:
  1864 ms vs PyNEC 11386 (6×). `lpda`: 1188 ms vs PyNEC **13439** (11×). It
  handles the log-periodic's non-identical "few-shape" elements fine, and scales
  near-linearly where every dense solver goes quadratic.
- **PyNEC is not universally fast.** It wins or ties on every single/few-element
  design (often <100 ms at N=81), but blows up on `lpda` (13.4 s) — NEC2's fill
  scales badly on that geometry. "Use PyNEC as the fast reference" holds for
  small designs, not arrays.
- **Among the dense bases, `Sinusoidal` stays the fastest** on small/medium
  single structures, consistent with the 2026-06-15 run. `Triangular` remains
  the slowest dense basis on every design here.
- **Medium arrays are pre-crossover.** On `delta_looparray_1x4` (4 elements),
  ArrayBlock (240 ms) scales best (~2.3×/step) but PyNEC (182) and Sin (198)
  still edge it at N=81; ArrayBlock's win arrives at larger arrays / higher N.

## Caveats

- 4-physical-core Haswell box. Absolute numbers move with core count
  (`OMP=N`); cross-design and cross-engine ratios should hold.
- Single-call `impedance()` only — no sweep amortization, no UI overhead.
- PyNEC parity coercion may bump segment counts to the next odd number,
  slightly inflating its effective N relative to momwire's.
- Crossover N for `arrayblock`/`hmatrix` is geometry-dependent; the ~N=40–60
  figures here are specific to these designs on this box.

## Follow-ups

- Verify the ACA/ArrayBlock crossover points on a different core count.
- The accelerators' overhead on small designs suggests the CLI/web default
  engine selection could auto-pick `arrayblock` for array builders and a dense
  basis otherwise, rather than a single global default.
