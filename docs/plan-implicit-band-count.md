# Plan: implicit (hideable) band count for multiband groups

## Problem

Multiband designs (`fandipole`, `trap_fan_dipole`, `twoband_fan_dipole`) expose
their per-band knobs as a repeating `bands` group. The number of band rows the
frontend renders is driven by a *separate* scalar param named by the group's
`repeat_count` (conventionally `n_bands`). The frontend reads that param's **live
value** from `currentValues` (`web/frontend/src/App.tsx`, ~line 776):

```js
const countRaw = values[item.repeat_count];           // values["n_bands"]
const count = typeof countRaw === "number" ? Math.round(countRaw) : 0;
```

So `n_bands` **must be a visible schema param**. If a fixed-count design hides it
(e.g. `twoband_fan_dipole` is structurally two dipoles and tried
`ui_params: { "n_bands": { "hidden": True } }`), the param drops out of the
schema → `values["n_bands"]` is `undefined` → `count = 0` → the band group renders
**zero rows**. That's why `twoband_fan_dipole` currently keeps a 1–2 `n_bands`
slider it doesn't really want.

## Goal

Let a fixed-count multiband design **pin its band count and hide the `n_bands`
slider**, while the band group still renders the correct number of rows. Designs
that genuinely vary their count (fandipole 1–5) keep the live slider unchanged.

## Approach — a fallback count on the group spec

Resolve the instance count with a fallback chain:

1. the **live value** of the `repeat_count` param, if it's a visible param
   (current behavior — keeps add/remove-band designs working), else
2. a **fixed count** carried on the group spec.

### Backend (`web/adapter.py` + `ParamGroupSpec`)

- Add `fixed_count: int | None` to `ParamGroupSpec` and its frontend serialization.
- In `_group_spec_from_default`, set `fixed_count` to the resolved `repeat_count`
  param's default value (e.g. `n_bands` = 2) when that param is hidden. (Setting
  it unconditionally is also fine — the frontend prefers the live value, so it's
  inert for visible-`n_bands` designs.)

### Frontend (`App.tsx`, ~line 776)

```js
const live = values[item.repeat_count];
const count =
  typeof live === "number" ? Math.round(live)
  : (item.fixed_count ?? instances.length);
```

No other change: the instances array is already preallocated to `max_repeats`,
and `Math.min(count, instances.length)` clamps correctly.

### Solve path (already fine)

The hidden `n_bands` value still reaches the solver: `_build_builder` seeds from
the full `default_params` (`n_bands` stays 2) and `build_wires` reads
`int(self.n_bands)`. No request field is needed.

## Tests

- Backend schema test: a builder with `ui_params: { "n_bands": { "hidden": True } }`
  produces a `bands` `ParamGroupSpec` whose `fixed_count == 2` and **no** `n_bands`
  scalar spec.
- Frontend (when there's a harness, or manual): the band group renders 2 rows with
  `n_bands` hidden.

## Then

Once shipped, set `twoband_fan_dipole`'s `n_bands` back to `{ "hidden": True }`
(its topology is fixed at two dipoles) to drop the vestigial slider.

## Scope

Small but cross-cutting — one dataclass field + one adapter line + one frontend
line + a test. Its own PR.
