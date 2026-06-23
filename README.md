# AntennaKNoBs &nbsp;·&nbsp; *by KK7KNB*

### Script your antenna. Tune it in real time by turning knobs.

AntennaKNoBs is a Python package for **parametric, programmatic antenna design**.
You describe an antenna once as a small Python *builder* — its geometry expressed
in terms of named parameters — and then explore the design space two ways:

- **In code**, from the command line or a Python script: draw geometry, sweep a
  parameter, compare radiation patterns, optimize for match or gain, export a
  NEC deck.
- **In the browser**, from a live workbench: drag a knob and watch the 3D wire
  model, far-field patterns, and Smith chart redraw in real time.

Its built-in engine is **momwire**, a new in-house set of method-of-moments
engines. You can *optionally* add **PyNEC** (the battle-tested NEC2 engine) as a
second backend and solve the same design both ways to trust the answer.

[![Test Python package](https://github.com/stevenmburns/antenna_designer/actions/workflows/test.yml/badge.svg)](https://github.com/stevenmburns/antenna_designer/actions/workflows/test.yml)
[![Ruff](https://github.com/stevenmburns/antenna_designer/actions/workflows/ruff.yml/badge.svg)](https://github.com/stevenmburns/antenna_designer/actions/workflows/ruff.yml)
[![Coverage](https://raw.githubusercontent.com/stevenmburns/antenna_designer/python-coverage-comment-action-data/badge.svg)](https://github.com/stevenmburns/antenna_designer/actions/workflows/test.yml)

---

## The live web workbench

The workbench is the fastest way to feel a design. Pick an antenna, and its
parameters appear as a panel of sliders — the *knobs*. Drag one and every view
updates live over a WebSocket: the solver re-runs and the browser redraws.

<!-- TODO: add a screenshot/gif of the web workbench here (web-workbench.png) -->

What you get:

- **A panel of knobs.** Every builder parameter becomes a slider (or dropdown,
  or checkbox) with sensible min/max/step. Drag and the design re-solves.
- **3D wire geometry** with current visualization, viewable from three
  orthogonal projections (top / front / side).
- **Azimuth and elevation** far-field pattern slices.
- **A Smith chart** of input impedance, with optional frequency-sweep and
  convergence overlays.
- **Three solver slots (A / B / C)** you can point at different backends and
  compare side by side — e.g. momwire triangular vs. B-spline vs. PyNEC on the
  same antenna, at once.

Live tuning stays responsive because rapid slider drags are coalesced into one
solve per round-trip, so the solver is never buried under stale requests.

### Running it

The workbench is a FastAPI backend plus a React (Vite) frontend. In development
you run the two together (two terminals):

```bash
# Terminal 1 — backend (from the repo root, in your .venv)
pip install -e ".[web]"
uvicorn web.server:app --reload          # serves on http://127.0.0.1:8000

# Terminal 2 — frontend dev server
cd web/frontend
npm install
npm run dev                              # open http://localhost:5173
```

The Vite dev server proxies the API and the `/ws` live-solve channel to the
backend on port 8000, so you only ever open `http://localhost:5173`.

For a production bundle, `npm run build` (output in `web/frontend/dist/`) and
serve it behind the FastAPI app.

> The `[web]` extra pulls in `uvicorn[standard]`, which includes the WebSocket
> support the live-solve channel needs — plain `uvicorn` fails the `/ws`
> handshake.

---

## Two simulation backends

AntennaKNoBs can solve any design with either backend, selected per-run with
`--engine` (CLI) or per-slot (web). Solving the same antenna two ways is the
point — agreement between independent engines is your confidence check.

| | **PyNEC** | **momwire** |
|---|---|---|
| What | Python binding to the compiled C++ **NEC2** engine | In-house **method-of-moments** engines, pure-Python core with optional C++ accelerators |
| Basis | NEC2 thin-wire (pulse/sinusoidal) | Multiple families: triangular (tent), sinusoidal, B-spline, H-matrix, array-block |
| Speed | Very fast single-frequency solves | Fast; C++ accelerators (pybind11) for assembly/quadrature, pure-Python fallback |
| Ground | Sommerfeld–Norton finite ground (default) | PEC image method; free space by default |
| Install | Prebuilt wheel from the `python-necpp` fork release (OpenBLAS vendored) | C++ accelerator built from the `momwire` submodule |
| Use it for | The established reference; finite-ground patterns | Basis-flexible cross-validation; geometries where NEC2 reactance fails to converge |

**Selecting an engine** (CLI):

```bash
--engine pynec                 # NEC2 via PyNEC
--engine momwire                 # momwire, default triangular basis
--engine momwire:triangular      # piecewise-linear (tent) basis  — the momwire default
--engine momwire:sinusoidal      # NEC2-style three-term basis (cross-validator)
--engine momwire:bspline         # degree-1/2 B-spline Galerkin basis
--engine momwire:hmatrix         # B-spline + hierarchical-matrix (ACA) acceleration
--engine momwire:arrayblock      # element-aware block solver for arrays
```

In Python, instantiate an engine directly:

```python
from antenna_designer.engines import PyNECEngine, MomwireEngine
from momwire import BSplineSolver

engine = PyNECEngine(builder)
engine = MomwireEngine(builder, solver=BSplineSolver, solver_kwargs={"degree": 2})
```

**momwire** lives in its own repository and is vendored here as a git submodule;
its primary `TriangularSolver` engine converges to NEC accuracy in ~80 segments
and is validated against the independent B-spline basis. The H-matrix and
array-block engines are newer and aimed at large arrays. **PyNEC** is an
*optional* second backend — the `python-necpp` fork, distributed as a
self-contained wheel (OpenBLAS vendored, so no SWIG/BLAS/autotools toolchain is
required at install time). It is licensed **GPL-2.0** and installed separately
from its own release; antenna_designer (MIT) neither bundles nor depends on it,
and loads it only if present.

---

## Designing antennas in code

An antenna is a subclass of `AntennaBuilder` that declares named parameters and
builds its wires from them. Because the geometry is *computed* from parameters
in ordinary Python, you specify physical coordinates a minimal number of times —
the rest follow by reflection and relative position. (Most antenna tools make
you type six absolute coordinates per wire.)

Here is the built-in Moxon beam (`beams.moxon`), abbreviated. Four parameters
describe the rectangle; helper functions negate coordinates (`rx`, `ry`) and
chain nodes into wires (`build_path`):

```python
from ... import AntennaBuilder
from types import MappingProxyType


class Builder(AntennaBuilder):
    default_params = MappingProxyType(
        {
            "freq": 28.57,
            "base": 7.0,
            "halfdriver": 2.4597430629596713,   # length of one radiating side
            "aspect_ratio": 0.3646010186757216,  # short side / long side
            "tipspacer_factor": 0.07729647745945359,
            "t0_factor": 0.4078045966770739,
        }
    )

    def build_wires(self):
        eps = 0.05
        base = self.base

        long = 2 * self.halfdriver / (1 + 2 * self.aspect_ratio * self.t0_factor)
        short = self.aspect_ratio * long
        tipspacer = short * self.tipspacer_factor
        t0 = short * self.t0_factor

        def build_path(lst, ns, ex):
            return ((a, b, ns, ex) for a, b in zip(lst[:-1], lst[1:]))
        def rx(p): return -p[0], p[1], p[2]   # mirror across x
        def ry(p): return p[0], -p[1], p[2]   # mirror across y

        S = (short / 2, eps, base)
        A = (S[0], long / 2, base)
        B = (A[0] - t0, A[1], base)
        C = (B[0] - tipspacer, B[1], base)
        D = rx(A)
        E, F, G, H, T = ry(D), ry(C), ry(B), ry(A), ry(S)

        n_seg0, n_seg1 = 21, 1
        tups = []
        tups.extend(build_path([S, A, B], n_seg0, None))
        tups.extend(build_path([C, D, E, F], n_seg0, None))
        tups.extend(build_path([G, H, T], n_seg0, None))
        tups.append((T, S, n_seg1, 1 + 0j))   # the driven segment
        return tups
```

The top-level package re-exports the workhorse functions, so a full
design-explore-compare loop is a short script. This optimizes an inverted-V
dipole at several heights and overlays the resulting patterns:

```python
import antenna_designer as ant
from antenna_designer.designs.dipoles.invvee import Builder

p = dict(Builder.default_params)
bounds = ((p['length_factor'] * .8, p['length_factor'] * 1.25), (0, 1))

builders = (
    ant.optimize(
        Builder(dict(p, base=base)),
        ['length_factor', 'angle_radians'], z0=50, bounds=bounds,
    )
    for base in [5, 6, 7, 8]
)

ant.compare_patterns(builders)
```

---

## Command-line usage

Everything is under `python -m antenna_designer <subcommand>`. Designs are named
`family.name` (with an optional `:variant`) — run `list` to see them all.

```bash
# Draw a Moxon's wire geometry to a file
python -m antenna_designer draw --builder beams.moxon --fn moxon.png

# Sweep frequency and plot impedance on a Smith chart
python -m antenna_designer sweep --builder beams.moxon --param freq \
    --use_smithchart --npoints 21 --fn moxon_smith.png

# Far-field pattern of a Yagi, solved with momwire
python -m antenna_designer pattern --builder beams.yagi --engine momwire:triangular

# Overlay patterns of three beams
python -m antenna_designer compare_patterns \
    --builders beams.moxon beams.hexbeam beams.yagi --fn beams.png

# Cross-check one design across two backends
python -m antenna_designer compare_patterns \
    --builders beams.moxon beams.moxon --engines pynec momwire:bspline --fn check.png

# Optimize length and arm angle of an inverted-V dipole for a 50 Ω match
python -m antenna_designer optimize --builder dipoles.invvee \
    --params length_factor angle_radians

# Export a NEC2 card deck for use in external tools
python -m antenna_designer export --builder beams.hexbeam --out hexbeam.nec

# List the available designs (optionally filter)
python -m antenna_designer list
python -m antenna_designer list dipole
```

Shared flags: `--engine` (backend, see above), `--ground`
(`free` | `pec` | `finite` | `finite:<eps_r>,<sigma>`), `--builder`/`--builders`,
and `--fn` (save to file instead of showing on screen).

Below is a typical far-field plot produced by the `pattern`/`compare_patterns`
commands:

![Radiation pattern](RadiationPattern.png)

### Available designs

Roughly 70 built-in designs across nine families — run
`python -m antenna_designer list` for the authoritative list:

| Family | Examples |
|---|---|
| `dipoles` | invvee, folded_invvee, ocf_dipole, koch_dipole, dipole_turnstile |
| `beams` | moxon, hexbeam, yagi, hb9cv |
| `loops` | quad, delta_loop, diamond_loop, horizontal_loop, bisquare |
| `verticals` | vertical, jpole, inverted_l, bobtail, four_square, bruce |
| `arrays` | yagiarray, moxonarray, invveearray, bowtiearray, delta_looparray |
| `multiband` | fandipole, trap_dipole, hexbeam_5band, twoband_fan_dipole |
| `broadband` | discone, g5rv, lpda, t2fd |
| `wire` | sterba, rhombic, vbeam, w8jk, zepp, lazy_h, longwire |
| `specialty` | hentenna, bowtie, helix, hourglass |

User-authored designs (in the `user.*` namespace) appear here too; filter with
`list --builtin-only` / `list --user-only`.

---

## Install

On recent Ubuntu (22.04 / 24.04). PyNEC installs as a prebuilt wheel, so no
SWIG/BLAS/autotools toolchain is needed; only the momwire C++ accelerator compiles
from source (hence `g++`).

**1. System dependencies**

```bash
sudo apt-get update
sudo apt-get install \
    python3 python3-pip python3-venv python3-dev \
    g++ build-essential git
```

**2. Clone and create a virtual environment**

```bash
git clone https://github.com/stevenmburns/antenna_designer
cd antenna_designer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install setuptools numpy scipy pytest matplotlib icecream scikit-rf
```

**3. Install momwire (the engine)**

```bash
# momwire: a git submodule; its C++ accelerator builds from source.
pip install pybind11
git submodule update --init momwire
pip install --no-build-isolation -e ./momwire
```

**3b. (Optional) Install PyNEC for cross-validation**

PyNEC is an optional second backend — **GPL-2.0**, installed separately from its
own release, and never bundled with or required by antenna_designer. Skip it and
momwire is still fully functional; install it only if you want to cross-check
against NEC2.

```bash
# The self-contained wheel from the python-necpp fork's release (OpenBLAS
# vendored). --no-index avoids upstream PyNEC on PyPI, whose builds are broken
# on current Python and lack the fork's BLAS/OpenMP work.
pip install PyNEC --no-index \
    --find-links https://github.com/stevenmburns/python-necpp/releases/expanded_assets/v1.7.4-accel.1
```

**4. Install AntennaKNoBs**

```bash
pip install -e .                 # or  pip install -e ".[web]"  for the web workbench
```

**5. Run the tests**

```bash
pytest -vv --durations=0 -- tests/
```

> The authoritative, always-tested version of this whole sequence is the CI
> workflow at [`.github/workflows/test.yml`](.github/workflows/test.yml) — it
> installs both engines and runs the suite on every push. If anything here
> drifts, that file is the source of truth.

---

## License

MIT — see [LICENSE](LICENSE).
