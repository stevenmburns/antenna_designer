# Python Code for Amateur Radio Antenna Design

Antenna design and optimization using the PyNEC library (https://github.com/tmolteno/python-necpp.git)

The compile and run:

1. Download and install PyNEC in a virtual environment
```bash
python3 -m venv .venv
.venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install numpy scipy pytest
pip install PyNEC
```
2. Tests can be run using:
```bash
pytest -vv --durations=0 -- tests/
```

