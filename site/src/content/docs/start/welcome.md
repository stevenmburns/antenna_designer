---
title: What is antennaknobs?
description: An open-source, browser-based wire-antenna MoM simulator — the tool, who it's for, and how the pieces fit together.
---

**antennaknobs** is an open-source simulator for wire antennas. You describe an
antenna as a set of wires, and a [method-of-moments](/reference/solver/) (MoM)
engine solves for the currents — giving you impedance, SWR, gain, and the
radiation pattern. The distinctive part is the **web workbench**: every design
exposes a panel of *knobs* (its parameters), and as you drag a knob the results
re-solve and redraw in real time.

## Who it's for

- **Ham operators & antenna builders** — size and tune a real build (a dipole,
  a Yagi, a loop) and read off the dimensions, watching SWR and pattern as you go.
- **RF / EE students & educators** — a hands-on way to *see* how geometry maps
  to impedance and radiation, with the source of every design open to read.
- **Python developers** — a small, clear framework for describing antenna
  geometry programmatically, designed so each built-in design is a readable
  starting point for your own.

## How the pieces fit

| Piece | What it is |
| --- | --- |
| **antennaknobs** | The Python package: the `AntennaBuilder` framework + the design catalog + the web workbench. |
| **momwire** | The in-house MoM engine (multi-basis, with H-matrix acceleration). Installed as a dependency. |
| **pynec-accel** | An optional NEC-2 backend used as a cross-validation reference. |

## Try it now

The live tuner is running here — open it, pick a design, and drag a knob:

👉 **[antennaknobs.fly.dev](https://antennaknobs.fly.dev/)**

Then come back for the [Quickstart](/start/quickstart/) to drive it from Python.
