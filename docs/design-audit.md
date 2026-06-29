# SPA design audit — AntennaKNoBs web workbench

Audit of the React/Vite single-page app (`src/antennaknobs/web/frontend/`):
`App.tsx` (~5.5k lines) + `styles.css` (~1.7k lines). Scope is **visual design,
UX, and accessibility** of the rendered chrome — not the canvas plot maths.

Date: 2026-06-28. Reviewer: design pass on `spa-design-audit`.

## Verdict

The SPA is **already well-built**, not a rough draft. It has a real,
token-driven design system (a single `:root` set drives both the DOM chrome and
the `<canvas>` plots), light/dark theming, a deliberate spacing/radius/type
scale, and genuine accessibility foundations (the rotary knob is a correct ARIA
`slider`; `prefers-reduced-motion` is honoured; focus-visible rings are
consistent; one contrast token was already bumped to 4.5:1). The findings below
are **refinements, not rescues** — most are small, and they cluster in the two
most recently-added surfaces (the reactive optimiser's knob-menu and
opt-control), which drifted from the system the rest of the app follows.

## Strengths (keep these)

- **One source of truth for theme.** `:root` tokens drive chrome + canvas; dark
  theme overrides *only* token values, so the whole app + plots retheme from one
  block. This is the right architecture and it's followed almost everywhere.
- **Scales are real and snapped.** `--space-1..7`, `--r-sm/md/lg`,
  `--text-lg/base/sm/xs` — ad-hoc px values were deliberately rounded into these.
- **Loading states are tasteful.** The solve bar is dwell-gated (≈300 ms) so
  quick solves never flash it, and stale panels dim with an eased transition.
- **The knob is accessible.** `role="slider"` with `aria-valuemin/max/now/text`,
  keyboard handling, and `aria-disabled`. Comboboxes use `role="combobox"` /
  `listbox` / `option` with `aria-expanded` and accessible names.
- **Motion is restrained and reversible**, with a global reduced-motion escape.

## Findings (prioritised)

### P1 — Accessibility

1. **Unlabeled `<select>`s.** Several selects have no accessible name (no
   wrapping `<label>`, no `aria-label`): the two band selects
   (`styles.css`-classed `band-select`, `App.tsx:3096` and `:3210`) and the
   optimiser objective select (`opt-objective`, `App.tsx:3243`). A screen reader
   announces them as an unnamed combobox. The antenna picker and variant select
   *do* set `aria-label`, so this is an easy, mechanical fix — add
   `aria-label="band"` / `aria-label="optimise for"`.
2. **Flat heading outline.** There is exactly one `<h1>` and **zero `<h2>`**.
   Section headers ("measurement freq", "simulation") render as
   `<div class="group-label">`, so assistive-tech heading navigation and the
   document outline see a single node. Promote group labels to `<h2>`/`<h3>`
   (style stays identical via the existing class) so the rail is navigable by
   heading.
3. **`title=` carries non-decorative information.** 28 `title` attributes hold
   real semantics — ground-model parameters (εr/σ), solver trade-offs, the
   "fast ground" caveat. `title` is invisible to keyboard and touch users and is
   announced inconsistently by screen readers. For the load-bearing ones (the
   ground/solver explanations), move the text into visible helper copy or an
   accessible popover; keep `title` only for redundant "Switch to X" hints.

### P2 — Design-system drift (consistency)

4. **Undefined radius token.** `.knob-menu` uses `border-radius:
   var(--radius-2, 6px)` (`styles.css:1591`) — `--radius-2` is **never defined**
   (the scale is `--r-sm/md/lg`), so it silently falls back to a hardcoded 6px.
   Should be `var(--r-sm)`.
5. **Off-scale elevation/shadows.** Three overlays hardcode black shadows
   instead of the theme-aware `var(--shadow)`: gear-menu
   (`rgb(0 0 0 / 0.28)`, `:340`), stage-readout (`rgba(0,0,0,0.28)`, `:1365`),
   knob-menu (`rgba(0,0,0,0.3)`, `:1592`). These won't track the light/dark
   `--shadow` value (light theme uses a blue-tinted `rgba(30,55,95,0.16)`).
   Consider a small elevation scale (`--shadow-1/2/3`) and route all overlays
   through it.
6. **Two inline `fontSize: 12` literals** (`App.tsx:3336`, `:3829`) bypass
   `--text-sm`. Minor, but they're the only type-scale escapes in the JSX.

### P2 — Optimiser UI integration

7. **Free-variable ring competes with the focus ring.** A knob marked as an
   optimiser variable gets `box-shadow: 0 0 0 2px var(--accent)`
   (`styles.css:1540`); the keyboard focus ring is `0 0 0 3px var(--accent-soft)`.
   When a free-variable knob is focused, the two rings stack and read as a muddy
   double outline. Differentiate the "is a variable" state from focus — e.g. a
   ring on the *cap* + the existing `⟳` glyph for the variable state, reserving
   the box-shadow purely for focus.

### P3 — Touch / responsive

8. **Sub-44px tap targets.** The header icon buttons (theme toggle, tools gear)
   are 28×28px and the band selects are compact; below ~44px they're fiddly on
   the phone layout the app explicitly supports (the `max-width:700px` pannable
   mode). Bump the icon buttons toward 36–40px on coarse pointers
   (`@media (pointer: coarse)`).
9. **Colour-only signal in the heatmap and some readouts.** The current-density
   ramp encodes magnitude by colour alone; the `val-hot` readout partly mitigates
   with `font-weight:600` but still leans on hue. Fine for the data viz, worth a
   note for the readouts — keep the weight/affix cue so SWR state isn't hue-only.

## Suggested order of work

1. The P1 a11y fixes (1–3) — small, high-leverage, and a few are one-line
   `aria-label`s.
2. The token-drift trio (4–6) — pure search-and-replace, zero behaviour change.
3. The optimiser ring (7) and touch targets (8) — small visual tweaks.

None of these are architectural. The system is sound; this is polish on a
mature surface.
