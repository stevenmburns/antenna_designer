# A/B live-compare — design notes & gotchas (parked)

Status: **parked in draft PR**. This was a first, minimally-invasive attempt at
"A/B live compare" — two independently-editable antenna configs, both solved
live, patterns overlaid. It works end-to-end (verified in-browser: dipole A vs
yagi B overlaid live, both editable, metrics table for both). But it revealed
that the frontend's single-active-config assumption runs deep, and the clean way
forward is a **state refactor first** (make "the current config" a first-class,
per-slot object) rather than bolting a second config alongside the first.

This doc records what was built and the gotchas, so the next attempt can decide
whether to build on the draft or restart after the refactor.

## What A/B live-compare is meant to be

Two configs (A and B), each its own design + variant + knobs + freqs, both
solved live, their radiation patterns overlaid on the same polar plots, with a
side-by-side metrics table. Distinct from the already-shipped **pin/ghost**
feature (PR #199), where B is a *frozen* snapshot; here B stays editable.

## What this attempt built

- **`POST /solve`** (`web/server.py`) — one-shot REST solve reusing the existing
  stateless `solve(req)` (same cache as `/ws`), so slot B can solve on demand
  without contending for slot A's live websocket. Errors returned in-band like
  `/ws`. Tested in `tests/test_web_server.py`.
- **`SlotBEditor`** (`App.tsx`) — a self-contained component holding B's own
  geometry / variant / knob values / freqs, rendering `GeometryCombobox` +
  variant `<select>` + `ParamForm` + freq inputs, and solving via `/solve`
  (debounced) plus `/pattern_metrics` for the table. Inherits A's solver +
  ground via a `solveBase` prop so the comparison is apples-to-apples.
- **Tabbed A|B UI** — a "Compare A/B" toggle + A|B tabs at the top of the
  sidebar. Tab A shows the existing sidebar; tab B shows `SlotBEditor`. Slot A's
  live WS loop is **untouched**.
- **Overlay + table** — `FarFieldChart` gained a `compare` prop drawn as a solid
  blue live trace (reusing the extracted `computeCutDbi`); `PatternCompareTable`
  gained an A/B row pair.

## Gotchas (the reason to refactor first)

1. **`paramValues` is keyed by geometry.** `Record<geometry, values>` can't hold
   *two different tunings of the same design*, which is a primary A/B use case
   (compare two tunings of one antenna). This forced B to carry fully
   independent value state. A refactor should make "config" a first-class object
   (`{geometry, variant, values, measFreq, designFreq, backend, ground, …}`)
   stored per slot, e.g. `configs: Record<SlotId, Config>` + `activeSlot`, so the
   same design can appear in two slots with different knobs.

2. **`GeometryCombobox` does not filter itself.** It renders whatever `groups`
   it's handed; the *parent* filters `geomGroups` by its own query string.
   Passing slot A's already-filtered `geomGroups` to slot B silently broke B's
   search box (typing filtered nothing). Fixed here by having `SlotBEditor`
   rebuild its groups from its own filter (reusing module-level `matchesQuery` /
   `familyOf` / `FAMILY_LABELS` / `familyRank`). A refactor should make the
   combobox self-filtering (take `examples` + a query, filter internally).

3. **Naming collision: "slot".** There is already an `activeSlot` / `slots`
   (A/B/C) concept — but for **solver backends** (triangular / bspline / pynec),
   not antenna configs. The compare feature's A/B tabs overload "A/B". Pick
   distinct names in the refactor (e.g. `configSlot` vs `solverSlot`).

4. **Solve was websocket-only.** Added `POST /solve` to give B an on-demand path.
   This part is clean and worth keeping regardless of the UI approach.

5. **Slot B must stay mounted across tab switches.** If `SlotBEditor` unmounts
   when you leave tab B, its edits reset from the `initial` seed on remount.
   Worked around by mounting it whenever compare is on and hiding it with
   `display:none` on tab A. A cleaner design lifts B's state into the parent so
   the editor is a pure view over per-slot state.

6. **Tabbed layout hides A's shared controls under tab B.** SIM / solver / ground
   live in slot A's body, hidden while editing B; B inherits them via
   `solveBase`. Acceptable for an MVP, but a lifted-state design (or a
   side-by-side layout) would expose both configs' shared controls cleanly.

## Recommended refactor before resuming

Make the active antenna config a first-class object and store it per slot:

```
type Config = { geometry; variant; values; measFreq; designFreq;
                backend; opts; groundEnabled; groundFast; … };
const [configs, setConfigs] = useState<Record<SlotId, Config>>(…);
const [activeSlot, setActiveSlot] = useState<SlotId>("A");
```

Then `buildRequest`, the sidebar controls, and the solve loop all read
`configs[activeSlot]`. A/B (and N-way) compare, "compare two tunings of the same
design", the self-filtering combobox, and a slot-agnostic `ParamForm` all fall
out of that, instead of being special-cased around slot B. The `POST /solve`
endpoint and the `computeCutDbi` / overlay / metrics-table plumbing from this
attempt carry over unchanged.

## Where the code is

Draft PR / branch `ab-live-compare`. See the diff for `web/server.py`
(`/solve`), `web/frontend/src/App.tsx` (`SlotBEditor`, `computeCutDbi` compare
overlay, `PatternCompareTable` B row, the A|B tab UI), `styles.css`, and
`tests/test_web_server.py`.
