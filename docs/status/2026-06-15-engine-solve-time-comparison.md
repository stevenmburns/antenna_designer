# 2026-06-15 — engine solve-time comparison (fandipole, trap_fan_dipole)

## Goal

Measure single-call `impedance()` solve time for five MoM engines across
three segmentation levels on two representative designs, with the same
BLAS/OpenMP threading configuration the interactive UI runs under. Result
is a baseline reference for which (engine, N) combinations are viable for
live tuning vs. batch-only.

## Script

`scripts/profile_compare_engines.py` — runnable as
`.venv/bin/python scripts/profile_compare_engines.py`.

Threading setup mirrors `web/server.py` and is applied **before** any
numpy/scipy/PyNEC import (each library snapshots its env at its own
import time and ignores later changes):

```
OPENBLAS_NUM_THREADS = 1                       # avoid contention with the PyNEC/MKL pool
OMP_NUM_THREADS      = physical_core_count     # /proc/cpuinfo parse, 4 on this box
MKL_NUM_THREADS      = physical_core_count
OMP_WAIT_POLICY      = PASSIVE                 # park between OMP regions
GOMP_SPINCOUNT       = 0                       # no busy-spin barrier
```

The physical-core detector counts unique `(physical id, core id)` pairs in
`/proc/cpuinfo`, falling back to `cpu_count() // 2` (HT-assumed) on failure.

Per cell the script does one off-band warm-up call, then mean wall-clock
across each design's target bands:

- fandipole: 14.300, 18.1575, 21.383, 24.97, 28.47 MHz (warm-up 13.0)
- trap_fan_dipole: 18.1575, 21.383, 24.97, 28.47 MHz (warm-up 17.0)

Geometry/Z size is identical across band frequencies for a given
(design, N), so per-call cost is essentially N-and-solver-only and the
mean hides one-shot jitter.

## Results

Mean ms across the design's target bands, with half-spread in parens.
4 physical cores, env settings as above.

**fandipole** (5 spokes × 2 arms, all variable-length wires):

| engine | N=21          | N=41          | N=81           |
|--------|---------------|---------------|----------------|
| Tri    | 122 (±  4)    | 443 (± 18)    | 2265 (± 80)    |
| Bs1    | 106 (±  4)    | 291 (± 20)    | 1119 (± 67)    |
| Bs2    | 141 (±  5)    | 402 (± 10)    | 1367 (± 13)    |
| Sin    |  32 (±  1)    | 168 (± 21)    |  891 (± 43)    |
| PyNEC  | 114 (±  7)    | 636 (±  9)    | 4736 (± 80)    |

**trap_fan_dipole** (2 spokes × 2 arms, adaptive per-wire segmentation):

| engine | N=21         | N=41         | N=81          |
|--------|--------------|--------------|---------------|
| Tri    |  15 (±  1)   |  27 (±  2)   |  79 (±  6)    |
| Bs1    |  26 (±  1)   |  51 (±  3)   | 112 (±  2)    |
| Bs2    |  31 (±  1)   |  62 (±  2)   | 144 (±  3)    |
| Sin    |   4 (±  0)   |   7 (±  0)   |  24 (±  1)    |
| PyNEC  |   6 (±  0)   |  20 (±  0)   |  87 (±  1)    |

Reproducibility: a second back-to-back run reproduced every cell within
~5%, and the tight ±-spreads confirm the measurements are stable.

## Headlines

- **Sinusoidal is the fastest pysim basis everywhere.** ~3–4× faster
  than Tri/Bs1/Bs2 on fandipole at every N, and ~3–4× faster than the
  next-fastest pysim basis on trap_fan_dipole too.
- **PyNEC scales worst on fandipole** — 114 → 4736 ms from N=21→81
  (~42×). On the smaller trap geometry PyNEC is competitive with Sin at
  every N (6 vs 4 ms at N=21; 87 vs 24 ms at N=81).
- **Triangular flips position between designs.** Slowest on fandipole
  (2265 ms at N=81), but the *fastest* pysim basis on trap_fan_dipole
  after Sin (15 ms at N=21). The trap design pins both feed and trap
  wires to 1 segment, which plays to Tri's strengths.
- **Bs1 < Bs2 on cost.** Modest gap (e.g. 1119 vs 1367 ms at fandipole
  N=81; 112 vs 144 ms at trap N=81). Pick based on convergence quality,
  not speed.

## Why the threading config matters

Initial run (system defaults, no env config) reported numbers up to ~10×
slower with much larger run-to-run spreads. Examples:

| | no env config | UI env config |
|---|---|---|
| fandipole Sin N=21    | 326 ms (±353) | 32 ms (±1) |
| trap PyNEC N=21       |  10 ms (±  0) |  6 ms (±0) |
| trap Sin N=81         | 117 ms (± 36) | 24 ms (±1) |

The big wins come from `OMP_WAIT_POLICY=PASSIVE` + `GOMP_SPINCOUNT=0`:
between OMP parallel regions the workers would otherwise busy-spin
~80 ms at every barrier on this CPU, which inflates wall-clock and adds
jitter as the spin overlaps differently with the Python serial work
between solves.

Without that pin, the numbers reflect a configuration the UI never runs
under — they're not the right baseline for "is this engine fast enough
for live tuning". The script above bakes the UI env in so future runs
stay apples-to-apples.

## Caveats

- 4-physical-core box (KBL-class). On a different core count, OMP=N
  changes the absolute numbers; cross-design and cross-engine ratios
  should hold.
- PyNEC parity coercion may bump segment counts up to the next odd
  number, slightly inflating its N relative to pysim's (small effect
  at these N values).
- Single-call `impedance()` only — no sweep, no UI overhead, no
  `run_in_threadpool` boundary. Real live-tick timing includes some
  Python/IO above these numbers.

## Follow-ups

- Verify on a different core count (e.g. CI box) that ratios hold and
  the absolute slowdown scales as expected.
- If trap_fan_dipole becomes the canonical interactive default, the
  numbers above suggest PyNEC at N=21–41 (6–22 ms) and Sin at any N up
  to 81 (4–24 ms) are both viable; fandipole at N=81 is batch-only
  territory for every engine.
