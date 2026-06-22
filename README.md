# Python-based Amateur Radio Antenna Design and Modeling Package

Front-end code for antenna design and optimization using the PyNEC library (https://github.com/tmolteno/python-necpp.git)

# Basic Usage:

To render (produce an image) of a moxon antenna, try:
```bash
python -m antenna_designer draw --builder moxon
```

To see the impedance as a function of the `halfdriver` parameter in the moxon model, try:
```bash
python -m antenna_designer sweep --builder moxon --param halfdriver
```
To compare the far-field patterns of a moxon and hexbeam, try:
```bash
python -m antenna_designer compare_patterns --builders moxon hexbeam 
```
# More Advanced Usage

Optimize the `length_factor` and `angle_radians` of an inverted-V dipole antenna as the height (`base`) is swept across several different values and then show the far-field plots for the different builds:
```python3
import antenna_designer as ant
from antenna_designer.designs.dipoles.invvee import Builder

p = dict(Builder.default_params)
bounds = ((p['length_factor']*.8, p['length_factor']*1.25), (0, 1))

builders = (
  ant.optimize(
    Builder(dict(p, **{'base': base})),
    ['length_factor', 'angle_radians'], z0=50, bounds=bounds
  ) for base in [5, 6, 7, 8]
)

ant.compare_patterns(builders)
```

# Antenna Builder Example
Here is a python subclass that can be used to design a moxon antenna.
In this design, there are four parameters that describe the shape of the rectangle. The parameter `halfdriver` is the length of one side of the radiating element, and this includes half of the long side of the rectangle and a segment on the short side. The parameter `t0_factor` is the fraction of the short side segment. `tipspacer_factor` is the fraction of the short side segment that separates the driver and the reflector. Finally, `aspect_ratio` describes the ratio of the short side to the long side.

There are three helper functions that manipulate nodes, and form wires between nodes. The `rx` and `ry` functions negate the x and y coordinates, respectively, and the `build_path` functions constructs wires by connecting consecutive nodes. These functions, and the general use of Python programming constructs, allow the designer to specify physical coordinates a minimal number of times, perhaps reducing errors and simplifying the design. Few of the nodes have their coordinates specified explicitly, then rest being defined through node negation, and relative position with respect to other nodes. Most antenna design systems require six absolute coordinates per wire.

```python3
from .. import AntennaBuilder
from types import MappingProxyType

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'freq': 28.57,
    'base': 7,
    'halfdriver': 2.460
    'aspect_ratio': 0.3646
    'tipspacer_factor': 0.0772
    't0_factor': 0.4078
  })

  def build_wires(self):
    eps = 0.05
	base = self.base

    # short = aspect_ratio*long
    # halfdriver = long/2 + short*t0_factor
    # halfdriver = long/2 + aspect_ratio*long*t0_factor
    # 2*halfdriver = long + 2*aspect_ratio*long*t0_factor
    # 2*halfdriver = long*(1 + 2*aspect_ratio*t0_factor)
    # long = 2*halfdriver/(1 + 2*aspect_ratio*t0_factor)

    long = 2*self.halfdriver / (1 + 2*self.aspect_ratio*self.t0_factor)
    short = self.aspect_ratio * long

    tipspacer = short * self.tipspacer_factor
    t0 = short * self.t0_factor

    def build_path(lst, ns, ex):
      return ((a,b,ns,ex) for a,b in zip(lst[:-1], lst[1:]))
    def rx(p):
      return -p[0],  p[1], p[2]
    def ry(p):
      return  p[0], -p[1], p[2]

    """
    D----------C   B-----A
    |                    |
    |                    |
    |                    |
    |                    |
    |                    |
    |                    |
    |                    S
    |                    |
    |                    T
    |                    |
    |                    |
    |                    |
    |                    |
    |                    |
    |                    |
    E----------F   G-----H
	"""

    S = (short/2,        eps,    base) 
    A = (S[0],           long/2, base)
    B = (A[0]-t0,        A[1],   base)
    C = (B[0]-tipspacer, B[1],   base)
    D = rx(A)
    E, F, G, H, T = ry(D), ry(C), ry(B), ry(A), ry(S)

    n_seg0, n_seg1 = 21, 1
      
    tups = []
    tups.extend(build_path([S,A,B], n_seg0, None))
    tups.extend(build_path([C,D,E,F], n_seg0, None))
    tups.extend(build_path([G,H,T], n_seg0, None))
    tups.append((T, S, n_seg1, 1+0j))

    return tups
```

# Install

To compile and run on recent ubuntu systems (22.04 and 24.04):

1. Download system dependencies
```bash
sudo apt-get update
sudo apt-get install \
    # Python dependencies
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    # C++ build deps for the pysim accelerator (PyNEC ships as a prebuilt wheel)
    g++ \
    build-essential \
    git
```

2. Create a virtual environment and collect python dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install setuptools numpy scipy pytest matplotlib icecream scikit-rf
```

3. Clone the repo and install the two engines (PyNEC + pysim). Do this in the virtual environment.
```bash
git clone https://github.com/stevenmburns/antenna_designer
cd antenna_designer

# PyNEC: install the self-contained wheel from the python-necpp fork's release
# (OpenBLAS is vendored, so no BLAS/SWIG/autotools toolchain is needed).
# --no-index avoids upstream PyNEC on PyPI, whose builds are broken on current
# Python and which lacks the fork's BLAS/OpenMP work.
pip install PyNEC --no-index \
    --find-links https://github.com/stevenmburns/python-necpp/releases/expanded_assets/v1.7.4-accel.1

# pysim (the MoM engine) is still a git submodule; its C++ accelerator builds
# from source (needs g++ + pybind11; install pybind11 if you haven't).
pip install pybind11
git submodule update --init pysim
pip install --no-build-isolation -e ./pysim
```

4. Install an editable version of this repo (inside virtual enviroment and in the cloned repository directory)
```bash
pip install -e .
```

5. Tests can be run (inside virtual enviroment and in the cloned repository directory) using:
```bash
pytest -vv --durations=0 -- tests/
```

There are dockerfiles to automate and document the necessary packages installations.

To build the docker image locally, try:
```bash
docker build -f Dockerfile --target antenna_designer --tag stevenmburns/antenna_designer .
```
To run some of the test in docker, try:
```bash
docker run -it stevenmburns/antenna_designer /bin/bash -c "source /opt/.venv/bin/activate && cd /opt/antenna_designer && pytest -vv --durations=0 -- tests/" 
```

The tests are also running in Github Actions on every push and pull request (PyNEC from the release wheel, pysim built from its submodule):

[![Test Python package](https://github.com/stevenmburns/antenna_design/actions/workflows/test.yml/badge.svg)](https://github.com/stevenmburns/antenna_design/actions/workflows/test.yml)

[![Ruff](https://github.com/stevenmburns/antenna_design/actions/workflows/ruff.yml/badge.svg)](https://github.com/stevenmburns/antenna_design/actions/workflows/ruff.yml)

[![Coverage](https://raw.githubusercontent.com/stevenmburns/antenna_designer/python-coverage-comment-action-data/badge.svg)](https://github.com/stevenmburns/antenna_design/actions/workflows/test.yml)

