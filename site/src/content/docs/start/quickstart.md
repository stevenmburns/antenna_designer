---
title: Quickstart
description: Install antennaknobs and solve your first antenna in a few lines of Python.
---

## Install

`antennaknobs` and its MIT engine `momwire` are published to PyPI with prebuilt
wheels — a plain install needs no compiler:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

pip install "antennaknobs[web]"
```

:::note
`momwire` (the solver) comes along as a dependency. The optional NEC-2
cross-validation backend `pynec-accel` is **GPL-2.0** — not a dependency of
antennaknobs — so install it on its own, never in the same command:
`pip install pynec-accel`. antennaknobs is fully functional without it.
:::

## Launch the web workbench

```bash
uvicorn antennaknobs.web.server:app      # then open http://127.0.0.1:8000
```

Pick a design from the dropdown and drag its knobs — the pattern, SWR, and
impedance re-solve live.

## Solve from Python

Every design is an [`AntennaBuilder`](/concepts/model/). Wrap one in an
`Antenna` and ask for its feed-point impedance:

```python
from antennaknobs import Antenna
from antennaknobs.designs.dipoles.invvee import Builder

ant = Antenna(Builder())     # an inverted-vee dipole, default parameters
print(ant.impedance())       # -> [(48.6-8.8j)]  ohms, one entry per feed port
```

Tune a knob and re-solve — parameters are plain attributes:

```python
b = Builder()
b.length_factor = 1.0        # stretch the arms
print(Antenna(b).impedance())
```

`Antenna` also gives you the far-field pattern, a frequency sweep of the
impedance, and the current distribution:

```python
ant.far_field()          # full-sphere far-field rings
ant.impedance_sweep(...)  # impedance across a frequency range
```

By default `Antenna` uses a finite ground; pass `ground="free"` (or a
`("finite", eps_r, sigma)` tuple) to change it.

:::tip[Next]
- [The model](/concepts/model/) — `build_wires()` and the knob system.
- [Many ways to express geometry](/concepts/authoring/) — the same loop, five ways.
- [Command line](/reference/cli/) — sweeps, patterns, and `.nec` export from the terminal.
:::
