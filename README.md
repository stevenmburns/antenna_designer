# Python Code for Amateur Radio Antenna Design

Antenna design and optimization using the PyNEC library (https://github.com/tmolteno/python-necpp.git)

The compile and run:

1. Download and install PyNEC in a virtual environment
2. Tests can be run using:
```bash
pytest -vv --durations=0 -- tests/
```

There are dockerfiles to automate and document the necessary packages installations.
To build the docker image locally, try:
```bash
docker build -f Dockerfile.local --target antenna_design --tag stevenmburns/antenna_design .
```
or, to use the cached environment container on docker hub:
```bash
docker build -f Dockerfile.dev --target antenna_design --tag stevenmburns/antenna_design .
```
To run some of the test in docker, try:
```bash
docker run -it stevenmburns/antenna_design /bin/bash -c "source /opt/.venv/bin/activate && cd /opt/antenna_design && pytest -vv --durations=0 -- tests/test_dipole.py tests/test_invvee.py" 
```


