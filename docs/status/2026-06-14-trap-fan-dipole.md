# 2026-06-14 — Adding the four-band trapped fan dipole design

## Goal

Add a four-band trap fan dipole (17m / 15m / 12m / 10m) to
`designs/freq_based/`. Two physical spokes from a common feed, each
broken by a parallel-LC trap. Spoke 0 covers 17m/12m; spoke 1 covers
15m/10m. PR [#69](https://github.com/stevenmburns/antenna_designer/pull/69).

## Final design summary

| Param | Value | Notes |
|---|---|---|
| `slope` | 0.68 | inverted-vee descent angle |
| `n_bands` | 2 | locked (UI exposes 1..2 but build_wires requires 2) |
| `trap0_freq_shift` | 1.0 | no shift (the `y==0` framework bug that forced 0.999 is now fixed — see follow-up) |
| `trap1_freq_shift` | 0.95 | puts 10m into a mode-jump regime where Re(Z)≈50 Ω |
| band 0 `full_freq` | 18.16 MHz | 17m |
| band 0 `trap_freq` | 24.97 MHz | 12m |
| band 0 `full_length_factor` | 0.4424 | half-arm / (λ/2 at 17m) |
| band 0 `inner_length_factor` | 0.4981 | half-arm / (λ/2 at 12m) |
| band 0 `trap_L_uH` | 3.0 | ~5–6 turns of 1 mm wire on a 15 mm form |
| band 0 `trap_C_pF` | 13.54 | LC-resonant at 24.97 MHz |
| band 1 `full_freq` | 21.38 MHz | 15m |
| band 1 `trap_freq` | 28.47 MHz | 10m |
| band 1 `full_length_factor` | 0.4532 | |
| band 1 `inner_length_factor` | 0.4474 | |
| band 1 `trap_L_uH` | 1.0 | ~4 turns of 1 mm wire on a 10 mm form |
| band 1 `trap_C_pF` | 31.25 | LC-resonant at 28.47 MHz |

Final SWR50 (BSplinePySim degree=2 @ nominal_nsegs=41):
- 17m (18.16 MHz): **1.03**
- 15m (21.38 MHz): **1.04**
- 12m (24.97 MHz): **1.05**
- 10m (28.47 MHz): **1.07**

Sensitivity to ±1 cm length error: max SWR ≤ 1.17 on any band.

## Key decisions and dead-ends

### Geometry

- **First cut copied fandipole's azimuth-grid construction** with
  `n = n_bands`. With n_bands=2 the grid degenerated — both spoke
  azimuths landed at cos=0, collapsing into the same x=0 plane.
  Comparison against fandipole's actual n_bands=2 output (it uses
  `n = _MAX_BANDS = 5` then slices) caught the divergence.
- **Per user direction, didn't restore the fandipole azimuth grid.**
  Instead replaced the cone with a horizontal-pigtail layout: each
  spoke originates from a short horizontal segment to (±radius, 0)
  before extending in the shared inverted-vee direction. This
  decouples `slope` from band-spacing — slope sweeps no longer
  perturb the inter-spoke separation near the feed.

### Trap inductance values

- **L1 (band-1 trap) started at 5 µH** (textbook default), which
  smothered 15m: the tank's below-resonance reactance at 15 MHz was
  +j1.4 kΩ, well past any compensation a length retune could deliver.
- **Dropped L1 to ~0.2 µH** to weaken the 15m loading. The smaller
  tank gave a max-SWR ≈ 1.13 design but was impractical (a few-turns
  coil whose inductance is dominated by stray/lead effects).
- **Settled on L1 = 1.0 µH** after a constrained scan — practical to
  wind, still keeps 15m loading manageable, max SWR ≈ 1.09.

- **L0 (band-0 trap) was pushed up to 11.85 µH** during early tuning
  on the assumption that heavier inductive loading would lower 17m's
  resonant Re(Z) toward 50 Ω. SWR50 at 17m dropped from ~1.31 (L=5)
  to ~1.27 (L=11.85), but...
- **±1 cm sensitivity analysis exposed it as unbuildable**: with
  L0=11.85, a 1 cm length error on band 0 sends 17m's SWR from 1.1
  to 3.4–5.3. The high-L operating point sits right at the cliff
  where the 17m series resonance is merging with the parallel-
  resonance pole — high Q → tiny tolerance in length-space.
- **Scanning L0 ∈ {3, 5, 8, 11.85} with full retune at each** showed
  L0=3.0 gives **better SWR (1.07 vs 1.09)** AND **dramatically
  better tolerance (1.17 vs 5.3)**. Settled there.

### Frequency-shift multipliers

- Added `trap0_freq_shift`, `trap1_freq_shift` as per-trap
  dimensionless multipliers on the tank's effective ω₀ (applied as
  C_eff = trap_C_pF / shift²). Default 1.0 = no shift.
- **trap1_freq_shift=0.95** was an important find: at this shift the
  band-1 trap is slightly past resonance at 10m, the outer extension
  joins the radiating section, and 10m's Re(Z) jumps from ~42 Ω to
  ~50 Ω — drops 10m SWR from 1.20 to 1.02.
- **trap0_freq_shift was briefly set to 0.999** as a defensive dodge of
  a framework bug in `network.load_impedance()` (see "Framework issues").
  That bug is now fixed, so the default is back to a clean 1.0 and band 0
  runs at exact LC-resonance.

### Choice of simulation basis

- Originally tuned against PyNEC at nominal_nsegs=21. Discovered
  later this was tracking a specific N's discretization error rather
  than the converged physical antenna.
- Ran a convergence sweep across all four bases (PyNEC, Pysim
  triangular, sinusoidal, bspline d=2) at N ∈ {11, 21, 41, 81}.
  **Bs2 (B-spline degree 2) was the only basis whose Z(N) stayed flat
  on the trap-loaded 17m and 10m bands** — PyNEC, triangular, and
  sinusoidal each drifted 20+ Ω across the N range there.
- Retuned everything against Bs2 at N=41. The result on PyNEC@21
  looks worse on paper (17m SWR ≈ 1.29 vs Bs2's 1.03) — but the
  engine-vs-engine spread on 17m exceeds any tuning we can do, so
  the right reference is the basis that converges.

### Performance fix discovered along the way

- Tuning loops were burning 80% of their time inside
  `get_elevation()` (full-sphere far-field integration) inside
  `opt.optimize()`'s objective. `get_elevation` was called
  unconditionally even when `opt_gain=False` and the result was
  thrown away.
- **Split into a separate branch `opt-skip-far-field`** (one commit)
  with the conditional `if opt_gain` gate. Will land as a follow-up
  PR after #69 merges. 5× speedup on resonance-only tuning.

## Framework issues / follow-ups

1. **`network.load_impedance()` y==0 guard** — FIXED (follow-up commit).
   The guard fired whenever a parallel-LC tank's L*C*ω² landed at
   *exactly* 1 in float arithmetic — i.e. the trap's design resonance,
   where Z=∞ (segment current interrupted, the open-circuit trap point).
   That is the *intended* operating point, not an error. Root cause was
   purely representational: the code formed Z_load = 1/y_tank and then
   the consumer formed 1/(1+Z_load·Y_kk), so the infinity appeared in an
   intermediate even though the final stamp is finite. Fix: added
   `load_series_admittance()` returning y_load = 1/Z_load directly (= the
   tank admittance, cleanly 0 at resonance), and reworked
   `PysimEngine._apply_loads` to the dual Sherman-Morrison form
   `Y -= outer / (y_load + Y_kk)` — algebraically identical for finite
   loads, finite at resonance (coefficient → 1/Y_kk). The defensive
   `trap0_freq_shift=0.999` was reverted to 1.0.

2. **`n_bands` schema test contract** — the `test_schema_covers_every_
   non_freq_default_param` and `test_param_schema_specs_are_typed_
   correctly` tests require that a tuple-of-dicts default
   (the `bands` group) has a corresponding `n_bands` scalar param
   with `min < max`. Our design is fixed at 2 spokes but the schema
   forced us to expose n_bands as a 1..2 range. n_bands=1 would
   trigger `build_wires`' explicit reject — not a great UX. Tighter
   schema semantics for "pinned" params would let designs lock a
   group size cleanly.

3. **The cross-engine impedance spread on trap-loaded bands** (PyNEC
   says 17m SWR ≈ 1.29, Bs2 says 1.03) is bigger than any tuning we
   can perform — for any real-world build, field tuning by
   measurement will be required regardless of which engine we
   trust. Worth flagging in user-facing docs once the design ships.

## What's *not* in this PR

- The `opt.py` far-field gate — separate branch `opt-skip-far-field`
  (merged as PR #70).
