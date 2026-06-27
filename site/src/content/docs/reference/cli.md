---
title: Command line
description: Driving antennaknobs from the terminal — list, draw, sweep, pattern, optimize, compare, and .nec export.
---

antennaknobs has a command-line interface for batch work. The subcommands:

```text
python -m antennaknobs {draw,sweep,optimize,pattern,compare_patterns,export,list}
```

| Command | What it does |
| --- | --- |
| `list` | List available designs (built-in and user) |
| `draw` | Draw the antenna geometry |
| `sweep` | Sweep a parameter or frequency |
| `pattern` | Plot the far-field pattern |
| `compare_patterns` | Overlay the patterns of several antennas / engines |
| `optimize` | Optimize an antenna's parameters |
| `export` | Export the design to a NEC-2 `.nec` card deck |

## Naming a design

Designs are addressed as `family.name` (the same names `list` prints):

```bash
python -m antennaknobs list            # arrays.bowtiearray, beams.yagi, loops.delta_loop, ...
```

## Patterns

```bash
# Far-field pattern of a Yagi, solved with momwire's triangular basis
python -m antennaknobs pattern --builder beams.yagi --engine momwire:triangular
```

Useful `pattern` flags: `--fn out.png` (write to a file instead of the screen),
`--ground free|pec|finite|finite:<eps_r>,<sigma>`, `--wireframe`, and
`--elevation_angle`.

## Choosing an engine

The `--engine` flag selects the solver:

```bash
--engine pynec                   # the NEC-2 reference backend (default)
--engine momwire                 # momwire, default (triangular) basis
--engine momwire:triangular      # piecewise-linear (tent) basis
--engine momwire:sinusoidal      # NEC-2-style three-term basis
--engine momwire:bspline         # B-spline Galerkin basis
```

See [The solver & accuracy](/reference/solver/) for which engine to reach for —
including the accelerated `hmatrix` and `arrayblock` solvers for large
single-wire structures and arrays.

## Comparing engines

Solve the same design two ways and overlay the patterns — the built-in
cross-validation:

```bash
python -m antennaknobs compare_patterns \
  --builders beams.moxon beams.moxon \
  --engines pynec momwire:bspline --fn check.png
```

## Exporting to NEC

```bash
python -m antennaknobs export --builder beams.yagi --fn yagi.nec
```

The deck is validated against `nec2c`, so designs round-trip into other NEC
tools.
