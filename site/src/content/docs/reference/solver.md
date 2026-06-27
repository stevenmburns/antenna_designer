---
title: The solver & accuracy
description: The momwire MoM engine — its basis functions, accelerated solvers, which one to pick, and how results are cross-validated.
---

antennaknobs solves antennas with **momwire**, an in-house method-of-moments
(MoM) engine for wire structures, with an optional NEC-2 backend for
cross-checking.

## Basis functions

momwire offers three current-expansion bases in one engine — uncommon among free
tools, where basis quality is usually a paid feature:

- **triangular** (piecewise-linear "tent") — the default,
- **sinusoidal** — a NEC-2-style three-term basis, useful as a cross-validator,
- **B-spline** — a degree-1/2 Galerkin basis.

Plus two **accelerated** solvers for large problems:

- **`hmatrix`** — ACA / hierarchical-matrix, sub-quadratic on large single-wire
  structures,
- **`arrayblock`** — element-aware block-low-rank, near-linear on arrays of
  identical / few-shape elements.

## Which solver should I use?

The choice turns on two axes — total problem size, and single-structure vs.
array geometry. From the [solver-selection
benchmark](https://github.com/stevenmburns/antennaknobs) (10 designs × 7
engines):

| Antenna class | Use | Why |
| --- | --- | --- |
| Single elements, small loops, beams, multiband dipoles | **`sinusoidal`** or **PyNEC** | the dense solve is milliseconds; the accelerators' setup is pure overhead here |
| Large single-wire structures (rhombic, long-wires, big loops) | **`hmatrix`** (ACA) | sub-quadratic scaling — the only solver that beats PyNEC on `rhombic` at high segmentation |
| Arrays of identical / few-shape elements (loop/bowtie arrays, LPDA) | **`arrayblock`** | 6–11× faster than PyNEC on large arrays; near-linear scaling |

:::caution
`arrayblock` / `hmatrix` only win at moderate-to-high segment counts on their
target geometries — below ~N=40 they're slower than a dense solve even there. Don't
reach for them on small problems.
:::

### What the numbers show

- **ACA earns its place on `rhombic`** — fastest engine at high segmentation
  (~3.9 s at N=81), beating PyNEC and every dense basis, scaling ~2×/step where
  dense solvers go ~5×/step.
- **ArrayBlock dominates arrays** — `lpda`: ~1.2 s vs PyNEC's **13.4 s** (11×);
  `bowtiearray2x4`: ~1.9 s vs 11.4 s (6×).
- **PyNEC is a great fast reference, but not universally fast** — it wins on
  single/few-element designs (often <100 ms), yet blows up on the log-periodic
  (13 s). "Use PyNEC as the reference" holds for small designs, not arrays.
- **Among dense bases, `sinusoidal` stays fastest** on small/medium single
  structures; `triangular` is the slowest dense basis.

## Accuracy & validation

- A NEC-2 reference engine (`pynec-accel`) runs alongside momwire, so any design
  can be solved two ways and compared — a built-in sanity check most tools lack.
- The repo carries the benchmark above plus per-design solver comparisons.

### Honest limitations

In the spirit of not overselling: momwire wires are currently PEC (no conductor
loss); finite-ground impedance uses an approximate image + Fresnel model rather
than full Sommerfeld (the NEC path covers that case); and the web UI renders
far-field as polar az/el cuts rather than a full 3D pattern surface (the solver
produces the full-sphere data — it's a rendering gap, not a solver one).

<!-- TODO: embed the benchmark plots once generated, and a parity/differentiator
     table vs PyNEC / 4nec2 / EZNEC / AN-SOF from the market-research doc. -->
