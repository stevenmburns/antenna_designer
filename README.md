# Python-based Amateur Radio Antenna Design and Modeling Package

Front-end code for antenna design and optimization using the PyNEC library (https://github.com/tmolteno/python-necpp.git)

# Basic Usage:

To render (produce an image) of a moxon antenna, try:
```bash
python -m antenna_designer draw --builder moxon
```

To see the impedance as a function of the `halfdriver` parameter in the moxon model, try:
```bash
python -m antenna_designer sweep --builder moxon --sweep_param halfdriver
```
To compare the far-field patterns of a moxon and hexbeam, try:
```bash
python -m antenna_designer compare_patterns --builders moxon hexbeam 
```
# More Advanced Usage

Optimize the `length` and `slope` of an inverted vee dipole antenna as the height (`base`) is swept across several different values and then show the far-field plots for the different builds:
```python3
import antenna_designer as ant
from antenna_designer.designs.invvee import Builder

p = Builder.default_params
bounds = ((p['length']*.8, p['length']*1.25),(0,1))

builders = (
  Builder(
    ant.optimize(
	  Builder(dict(p, **{'base': base})),
              ['length','slope'], z0=50, bounds=bounds
    )
  ) for base in [5,6,7,8]
)

ant.compare_patterns(builders)
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
    # C++ Dependencies
    g++ \
    gfortran \
    swig3.0 \
    git \
    build-essential \
    automake \
    libtool
```

2. Create a virtual environment and collect python dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install setuptools numpy scipy pytest matplotlib icecream
```

3. Clone repo, update the submodules (for python-necpp) and install from sources (can't do just a pip install PyNEC). Do this in the virtual environment.
```bash
git clone https://github.com/stevenmburns/antenna_design
cd antenna_design
git submodule init
git submodule update --remote
pushd python-necpp
git submodule init
git submodule update --remote
cd PyNEC
ln -s ../necpp_src .
pushd ../necpp_src
make -f Makefile.git
/configure --without-lapack
popd
swig3.0 -Wall -v -c++ -python PyNEC.i
python setup.py build
python setup.py install
popd
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
docker build -f Dockerfile --target antenna_design_base --tag stevenmburns/antenna_design_base .
```
To run some of the test in docker, try:
```bash
docker run -it stevenmburns/antenna_design_base /bin/bash -c "source /opt/.venv/bin/activate && cd /opt/antenna_design && pytest -vv --durations=0 -- tests/test_dipole.py tests/test_invvee.py" 
```

The tests (including a build of python-necpp from sources) are also running in Github Actions on every push and pull request:
[![Test Python package](https://github.com/stevenmburns/antenna_design/actions/workflows/test.yml/badge.svg)](https://github.com/stevenmburns/antenna_design/actions/workflows/test.yml)

[![Ruff](https://github.com/stevenmburns/antenna_design/actions/workflows/ruff.yml/badge.svg)](https://github.com/stevenmburns/antenna_design/actions/workflows/ruff.yml)

