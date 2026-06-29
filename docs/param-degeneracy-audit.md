# Parameter degeneracy audit — suppressable design knobs

Goal: find user-facing design parameters that can be hidden from the UI (via
`ui_params[<param>] = {"hidden": True}`) because they are **redundant** — either
two knobs expressing one degree of freedom (a "product degeneracy") or a param
not used at all. All 73 designs were audited (build_wires + helpers + default_params).

Precedent: `loops/bisquare` had `side = side_frac × wavelength × length_factor`,
so `side_frac` and `length_factor` were one DOF as two knobs; `side_frac` is now
hidden. (Done.)

## Selection rule for which knob to keep

When `dim = <frac> × wavelength × length_factor` is the only use of both params,
**keep the param that already carries curated UI bounds in `ui_params`; hide the
other** (the un-curated duplicate). This preserves the designer's intended tuning
range without transferring bounds. Where both carry bounds, keep `length_factor`
(the conventional scale/optimiser knob, consistent across the catalog).

## High-confidence — product-degenerate, recommend hiding (6)

Each verified by grep: both params appear on exactly one line, nowhere else.

| Design | Line | Hide | Keep | Why keep |
| --- | --- | --- | --- | --- |
| `loops/horizontal_loop` | `side = side_frac × wl × length_factor` | `side_frac` | `length_factor` | length_factor has bounds 0.9–1.1 |
| `dipoles/koch_dipole` | `span = span_frac × wl × length_factor` | `span_frac` | `length_factor` | both bounded; keep the scale knob (0.85–1.15) |
| `specialty/helix` | `axial = axial_frac × wl × length_factor` | `axial_frac` | `length_factor` | length_factor has bounds 0.7–1.3 |
| `wire/longwire` | `length = length_frac × wl × length_factor` | `length_frac` | `length_factor` | length_factor has bounds 0.9–1.1 |
| `broadband/g5rv` | `top = top_frac × wl × length_factor` | `length_factor` | `top_frac` | **top_frac** is the curated knob (1.0–2.0); length_factor is bare |
| `wire/zepp` | `length = length_frac × wl × length_factor` | `length_factor` | `length_frac` | **length_frac** is the curated knob (0.40–0.55); length_factor is bare |

This removes one redundant knob from each of these 6 designs (7 with bisquare).

## Medium-confidence — degenerate, but worth an eyeball (2)

- **`broadband/lpda`** — `lam_high = c/(freq_high_factor × design_freq)`, then
  `l_min = 0.5 × lam_high × length_factor`, so element scale ∝
  `length_factor / freq_high_factor`. A quotient, not a clean product, and
  `freq_high_factor` carries band-edge meaning (it defines the LPDA's high-freq
  corner). Mathematically one scale DOF; recommend hiding `length_factor`, keeping
  `freq_high_factor` — but confirm the band-edge knob is what users want to drive.
- **`verticals/four_square`** — `elem = elem_frac × wavelength × length_factor`,
  mechanically degenerate, but the docstring frames `elem_frac` (electrical type,
  ~0.5) and `length_factor` (resonance trim, 0.8–1.2) as conceptually separate.
  Recommend hiding `elem_frac` (pin 0.5), keeping `length_factor` — eyeball first.

## Low-confidence — flagged, do NOT suppress without investigation (2)

- **`del_z` in the four `Array1x2Builder` arrays** (`bowtiearray1x2`,
  `delta_looparray`, `hentenna_array`, `hourglass_array`) — referenced in
  `builder.py`, but a 1×2 array has a single element row, so `del_z` is a rigid
  z-translation of the whole antenna; in free space (no ground/image) it's
  EM-inert. Default 0.0. Possibly a dead knob, but it IS referenced — confirm no
  solve path introduces a ground plane before hiding.
- **`freq_10` / `freq_12` in `multiband/twoband_fan_dipole`** — never read by
  `build_wires` (only printed in `__main__`). Almost certainly per-band
  measurement-frequency anchors consumed by the solver harness, not geometry —
  confirm against the sweep/meas-freq machinery before treating as dead.

## Clean — no suppressable params (the rest)

The other ~60 designs are clean. Notable non-degeneracies the auditors confirmed:

- **The `length_factor` + `angle_deg` pattern** (delta_loop, invvee,
  folded_invvee, yagi, and their arrays) is genuinely two DOF — one scales length,
  the other sets droop/spread via sin/cos — never multiplied together.
- **The sterba family** (`0.5 × wavelength × length_factor`) is NOT degenerate:
  the coefficient is a constant, and `length_factor` independently drives the wire
  geometry. `sterba_difftl.tl_length_factor` is a real independent phase-trim DOF.
- **Multi-`_factor` leaves** (hentenna, hourglass, moxon, hexbeam) each use every
  factor independently somewhere, even where some share a sub-expression.
- **Per-band `length_factor`/`freq` pairs** in multiband designs pair the factor
  with an independently-meaningful band `freq`, so they're not redundant.

## Net

7 designs collapse from 2 knobs → 1 (bisquare done; 6 more high-confidence).
2 medium-confidence candidates pending a design call. 2 low-confidence items need
a code check (ground handling; meas-freq harness). No widespread dead knobs.
