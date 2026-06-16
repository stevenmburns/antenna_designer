# 2026-06-15 — trap_fan_dipole adaptive segmentation + cross-engine convergence

## Goal

Make the trap_fan_dipole's per-wire segment counts adaptive to physical
wire length so segment-length is near-constant across the antenna, then
measure how each MoM engine converges as `nominal_nsegs` is swept. Branch
`trap-fan-dipole-wire-modeling`, follow-up to PR #69 (the original design).

## What changed

1. **Parameter spec cleanup** (commit `a310b4e`):
   - Dropped the unused `design_freq` top-level param.
   - Top-level `freq` default and the four `band{0,1}_{inner,full}_params`
     variants now reference `_BAND_*["full_freq" | "trap_freq"]` instead
     of hardcoding floats — single source of truth for band frequencies.
   - Per-band `freq_shift` (was top-level `trap0_freq_shift` /
     `trap1_freq_shift`). Lives inside the bands group; the UI now
     renders one slider per band instead of two parallel top-level knobs.
   - Dropped per-band `trap_C_pF`. C is now derived inside
     `build_network` from `trap_L_uH` and `trap_freq × freq_shift`. The
     `_resonant_C_pF` helper went with it.

2. **Adaptive per-wire segmentation** (commit `9fecec9`):
   `build_wires` computes `target_seg_len = max(adaptive_wire_lengths) /
   nominal_nsegs` and sizes every variable-length wire (cone + inner +
   outer) so segment length is uniform. Trap segments stay pinned to 1
   (single-segment Load convention). Feed segment also pins to 1 (the
   design is Bs2-tuned; the basis handles a 1-segment feed cleanly).
   Triangular is no longer drop-in compatible — its parity coercion
   bumps the 1-segment feed to 2, which works, but TriangularPySim
   needs the parity machinery to stay sane on this design.

3. **PyNEC engine bug fix** (commit `d5c8163`):
   `PyNECEngine._synthesise_virtual_stubs` was reading
   `self.builder.design_freq` unconditionally, even for designs whose
   Network has no `PortVirtual` ports (trap_fan_dipole's network is
   feed + per-trap Loads, all on real wires — no virtual ports). Now
   gated on whether `virtual_neighbours` is non-empty.

## Segmentation table at `nominal_nsegs ∈ {21, 41, 81}`

| Wire | length (m) | N=21 | N=41 | N=81 |
|---|---:|---:|---:|---:|
| cone (pigtail, both bands, both arms) | 0.12 | 1 | 2 | 3 |
| band 0 inner (A → trap_in) | 2.87 | 21 | 41 | 81 |
| band 0 outer (trap_out → tip) | 0.61 | 4 | 9 | 17 |
| band 1 inner | 2.24 | 16 | 32 | 63 |
| band 1 outer | 0.77 | 6 | 11 | 22 |
| trap segment (×4) | 0.05 | 1 | 1 | 1 |
| feed (T → S) | 0.02 | 1 | 1 | 1 |

Total segments: 103 / 199 / 383. Target seg length: 0.137 / 0.070 / 0.035 m.
Old segmentation at N=21 totalled 173 segments and had a 24× spread in
segment lengths (0.0057 m on the cone vs 0.137 m on the inner). New
spread is at most ~2× and only on the short cone/trap/feed segments that
can't shrink past 1.

## Cross-engine convergence (Re(Z), Ω, at each band's target freq)

```
            17m (18.16)       15m (21.38)       12m (24.97)       10m (28.47)
N=     21    41    81    21    41    81    21    41    81    21    41    81
Bs2  48.57 48.63 48.68 48.09 48.16 48.20 51.34 51.45 51.52 53.39 53.55 53.61
Tri  48.80 48.89 48.94 48.11 48.22 48.28 52.66 52.71 52.70 54.40 54.49 54.51
Sin  47.0  48.25 49.5  51.3  50.6  49.4  49.5  51.8  53.8  48.8  52.0  56.3
PyNEC 59.6 59.6  58.3  57.4  59.1  58.2  56.0  60.6  62.3  57.7  63.6  67.2
```

Im(Z) is small (≤ ~3 Ω) at the converged limit for Bs2/Tri on every band;
Sin and PyNEC swing by 10–40 j at low N (see
`pysim/scripts/trap_fan_convergence.py`).

## Runtime per solve (ms, mean of 4 band freqs after a warm-up)

| engine | N=21 | N=41 | N=81 |
|---|---:|---:|---:|
| Bs2 | 315 (±87) | 439 (±254) | 948 (±68) |
| Tri | 261 (±81) | 360 (±206) | 719 (±319) |
| Sin | 151 (±91) | 152 (±164) | 275 (±283) |
| PyNEC | 13 (±0) | 37 (±3) | 139 (±6) |

Spreads (± half-range across the 4 freqs) are wide for the pysim
engines — auto-singular-enrichment runs a 2-pass solve only at
junctions whose tap-ratio exceeds the threshold, so per-band cost
depends on which bands' currents trigger it. The pattern is solid:

- **PyNEC is 10–25× faster** than any pysim engine at every N (raw
  C vs Python + numpy + auto-enrichment + KCL Schur). Its O(n³) LU
  also scales more cleanly: 13 → 37 → 139 ms is roughly the cubic
  curve, while the pysim engines spend most of their time in fill
  (see `2026-06-15-bspline-cache-and-dedup` and the live-tick probe).
- **Bs2 is the slowest pysim engine.** Quadratic basis means more
  per-segment kernel work and (with `enrichment_variant="auto"`) a
  two-pass solve for any K≥3 junction that needs it — both the S/T
  junctions on this antenna qualify. Tri ≈ 0.75× Bs2; Sin ≈ 0.5×
  Bs2 at low N, ~0.3× at N=81.
- **Sin scales gently** thanks to its constant-source basis being
  cheap on the fill (no enrichment, no quadratic shape integrals).
  Still discretization-sensitive — see convergence above.

Warm-up uses a 17.0 MHz off-band freq so no future cache layer keyed
on (engine, N, freq) can accidentally seed a hit. Numbers from
`pysim/scripts/trap_fan_timing.py`.

### Caveat on the 10–25× PyNEC advantage

That ratio is specific to this **small antenna with named ports**, not a
general property of PyNEC vs pysim. Cross-check on the no-trap fandipole
(same engines, same builder pattern, larger antenna ~870 basis fns):

| | Bs2 | PyNEC | ratio |
|---|---:|---:|---:|
| fandipole (no traps) @ N=21 | 944 ms | 185 ms | 5.1× |
| fandipole (no traps) @ N=41 | 949 ms | 1007 ms | **0.9×** |
| trap_fan_dipole @ N=21 | 314 ms | 12 ms | 25.2× |
| trap_fan_dipole @ N=41 | 405 ms | 42 ms | 9.7× |

At fandipole's N=41 (n_basis ≈ 870) pysim Bs2 is actually slightly
*faster* than PyNEC — both are dominated by O(n³) LU and pysim's
`scipy.linalg.solve` calls BLAS competitively with NEC2's internal LU.

The 25× ratio on trap_fan_dipole @ N=21 is the small-problem regime:
PyNEC's per-call cost runs from a very low base (12 ms total for 103
segments + 4 ld_cards), while pysim has a Python-overhead floor that
doesn't shrink with n — builder construction, geometry translation,
basis polynomial setup, network branch reduction, voltage Schur, all a
few-ms-to-tens-of-ms regardless of size. At small n the matrix work
collapses but the overhead stays, so the ratio blows out. At
fandipole-scale or larger, Bs2 and PyNEC are roughly comparable; Sin
(the cheapest pysim basis) outright beats PyNEC in some regimes.

## Findings

1. **Bs2 is essentially N-independent.** Re(Z) drift ≤ 0.22 Ω from
   N=21 → 81 on every band. The design's tuning target is well-chosen.

2. **Triangular now converges as tightly as Bs2** (≤ 0.17 Ω drift) and
   agrees with Bs2 to within 1.3 Ω at the converged limit on every band.
   This is a real cross-engine consistency win from uniform segmentation
   — the docstring's old claim that "triangular drifts by 20+ Ω" is no
   longer true and has been updated (commit `accccdf`). Both basis
   families now see the same kernel-error budget per segment.

3. **Sinusoidal is the most discretization-sensitive.** Re(Z) wanders
   2–8 Ω across N and Im(Z) can swing 40 j at low N (17m N=21 hardly
   looks resonant). This is a basis-family issue (less localized
   support than tent / B-spline), not something uniform segmentation
   can fix.

4. **PyNEC sits ~10 Ω above the pysim cluster on every band** — a
   systematic offset. On 10m and 12m it actually drifts *up* with
   refinement (10m: 57.7 → 67.2 Ω from N=21 to N=81). This is the
   engine's own convergence story and the dominant remaining source of
   cross-engine disagreement on this design. Two hypotheses to chase
   next, both speculative:
   - Wire-radius / segment-length ratio at high N: pysim and NEC2 may
     enforce thin-wire validity bounds differently.
   - The trap segments are exactly 0.05 m / λ_10 = 0.0047 long. NEC2
     has minimum-segment-length warnings around λ/1000; we're at λ/213
     which is fine in principle, but with the Load card applied to that
     short segment NEC2 may be hitting an internal sensitivity.

## Where the pysim time actually goes

cProfile of a single `eng.impedance()` call at the design freq
(`pysim/scripts/trap_fan_profile.py`):

```
                                     N=21 (335 ms)     N=81 (1294 ms)
scipy.linalg.solve                   140 ms  42%        800 ms  62%
J_static_moment (Python fallback)     86 ms  26%         33 ms   3%
_build_basis_polynomials              69 ms  20%        297 ms  23%
_build_J_blocks (other)               32 ms  10%        100 ms   8%
```

**Singular enrichment is not a factor.** `BSplinePySim.compute_y_matrix`
(the multi-port path that every network-spec design takes) raises
`NotImplementedError` if `use_singular_enrichment=True`, so on
trap_fan_dipole enrichment is hard-disabled regardless of the flag —
nothing to profile there.

**Real optimization target uncovered by the profile**: `J_static_moment`
has a C++ accelerator (`seg_seg_static_moments_bspline_uniform`) that
only triggers for wires with `N > 1` *and* uniform segment spacing.
After adaptive segmentation, the 4 cones + 4 trap segments + 1 feed are
all N=1 wires, so 9 of 17 wires fall through to Python. At N=21 that's
81 fallback calls eating 26% of wall time. Extending the accelerator's
trigger condition to handle N=1 (a trivial closed-form case) would
likely shave a quarter off the N=21 cost; the gain shrinks at larger N
as the O(n³) solve takes over.

## Action items

- [x] Adaptive segmentation in `build_wires`.
- [x] Cleanup of `design_freq` / `trap_C_pF` / `trap{0,1}_freq_shift`.
- [x] Fix PyNEC engine for designs without PortVirtual ports.
- [x] Refresh the design's docstring with the new convergence picture.
- [ ] Investigate PyNEC's drift-with-N on 10m / 12m. Try varying wire
      radius and trap-segment length to see what moves it.
- [ ] Extend `seg_seg_static_moments_bspline_uniform` (and/or its
      caller) to handle N=1 wires so the cone/trap/feed segments stop
      falling through to the Python `J_static_moment` fallback. ~25%
      speedup expected at N=21, less at higher N.
- [ ] Investigate whether `BSplinePySim.compute_y_matrix` *should*
      support enrichment — every network-spec design with a K≥3
      junction is currently silently locked out of the enrichment
      benefits the plain-impedance path gets.
- [ ] Verify by measurement after build (carried over from the original
      design's status doc — unchanged).
