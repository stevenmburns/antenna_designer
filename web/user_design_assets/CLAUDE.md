# Designing antennas in this folder

This folder holds **your** antenna designs for the Antenna Designer app.
Each `.py` file here is one antenna. Drop a file in, refresh the web page,
and it shows up under **"Your designs"**.

If you're reading this as Claude Code: the user wants you to write or edit an
antenna design in this folder. Follow the contract below exactly, then tell
them to refresh the page (and to check the "failed to load" panel if their
design has an error).

## The contract

A design file must:

1. **Be named** `lowercase_with_underscores.py`. The file name becomes the
   antenna's name in the app (`my_dipole.py` → `user.my_dipole`). No spaces,
   no dots except the `.py` extension, one antenna per file.
2. **Define a class named exactly `Builder`** that subclasses `AntennaBuilder`.
3. **Import only** from `antenna_designer` and the Python standard library.
   Do **not** import other design files — keep each one self-contained.
4. Provide a **`default_params`** mapping (use `MappingProxyType(...)`) and a
   **`build_wires(self)`** method.

Start from `TEMPLATE.py` in this folder — copy it, rename it, edit it.

The same file also works from the command line: once `my_dipole.py` is in
this folder, run e.g. `antenna_designer draw --builder user.my_dipole` (or
`sweep`, `pattern`, …). Always address it with the `user.` prefix.
`antenna_designer list` shows every available design name (yours included).

## `default_params`

Every key becomes a slider in the UI, accessed in `build_wires` as
`self.<key>`. Conventions:

- `freq` — measurement frequency in **MHz**. Always include it.
- Lengths/positions are in **metres**.
- Add a nested `"ui_params": MappingProxyType({...})` for UI hints. The most
  useful is `"default_view"`: `"xy"` (top-down), `"xz"`, or `"yz"` (side).
- A class-level `label = "Pretty Name"` sets the display name (optional).

Slider bounds and step are auto-derived (±50% around the default, fine
resolution). You usually don't need to specify them.

## `build_wires(self)`

Return a list of straight wire segments. Each entry is:

```python
(start, end, n_segments, feed)
```

- `start`, `end` — `(x, y, z)` tuples in metres.
- `n_segments` — how finely to subdivide that wire. Use `self.nominal_nsegs`
  for the main radiator; fewer for short stubs (`max(1, self.nominal_nsegs // 7)`).
- `feed` — `1 + 0j` on the **single** segment the transmitter drives, `None`
  everywhere else. Exactly one segment in the whole antenna is the feed.

**Feed convention:** put the feed on a tiny segment between two points a
small `eps` (e.g. 0.01 m) apart, with the radiator arms running outward from
those two points. Wires connect where they share an endpoint, so the arms and
the feed segment must share their centre points exactly. See `TEMPLATE.py`.

**Frequency-scaled geometry (optional):** to size an antenna from a design
frequency instead of fixed metres, add a `design_freq` param (MHz) and a
`length_factor` (a multiplier near 1.0), then compute
`wavelength = 299.792458 / self.design_freq` and build dimensions as
fractions of `wavelength * self.length_factor`. This is how most built-in
designs work and gives you a clean per-band tuning knob.

## Arrays of identical elements (advanced)

For a phased array of N copies of one element, `antenna_designer` exposes
`Array1x2Builder`, `Array1x4Builder`, `Array2x2Builder`, `Array2x4Builder`.
These wrap an element `Builder`. For a first design, stick to a single
`build_wires`; reach for arrays only once a single element works.

## How to check your work

1. Save the file. **Refresh the web page** (no server restart needed).
2. The antenna appears under "Your designs". If it doesn't, look at the
   **"designs that failed to load"** panel — it shows the file, the error
   type, and the line number. Fix and refresh again.
3. Pick it in the selector and look at the geometry plot, the SWR curve, and
   the impedance. Adjust `default_params` and `build_wires` until the
   resonance and pattern look right.

## Good prompts to give Claude Code

- "Make me a 40-meter off-center-fed dipole fed 1/3 from one end."
- "Design a 2-element 20-meter quad loop, driven element plus a reflector."
- "Take my_dipole.py and add an adjustable height-above-ground slider."
- "My design loads but resonates at 32 MHz — shorten it to hit 28.5 MHz."
