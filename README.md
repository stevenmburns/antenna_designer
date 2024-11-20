# Python Code for Amateur Radio Antenna Design

Antenna design and optimization using the PyNEC library (https://github.com/tmolteno/python-necpp.git)

The compile and run:

1. Download and install PyNEC in a virtual environment
2. Tests can be run using:
```bash
pytest -vv --durations=0 -- tests/
```

There is a `Dockerfile` to automate installation and document the necessary packages.
To build the docker image, try:
```bash
docker build --target antenna_design --tag stevenmburns/antenna_design .
```
To run some of the test in docker, try:
```bash
docker run -it stevenmburns/antenna_design /bin/bash -c "source /opt/.venv/bin/activate && cd /opt/antenna_design && pytest -vv --durations=0 -- tests/test_dipole.py tests/test_invvee.py" 
```


