#
# Base container starts here
#
FROM ubuntu:24.04 AS antenna_designer_base

# Update packages
RUN apt-get -qq update && DEBIAN_FRONTEND=noninterative apt-get -qq install \
    # Python dependencies
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    # C++ build deps for the pysim accelerator (PyNEC ships as a prebuilt wheel)
    g++ \
    git \
    build-essential \
&&    apt-get -qq clean

FROM antenna_designer_base AS antenna_designer_base_with_python

# Create virtual environment and install python packages
# Upgrade pip & install testing dependencies
# Note: Package dependencies are in setup.py
RUN \
    /bin/bash -c "python3 -m venv /opt/.venv && \
    source /opt/.venv/bin/activate && \
    pip install --upgrade pip -q && \
    pip install setuptools numpy scipy pytest matplotlib icecream scikit-rf coverage pybind11 fastapi httpx -q"

FROM antenna_designer_base_with_python AS antenna_designer

COPY . /opt/antenna_designer

RUN \
    /bin/bash -c "source /opt/.venv/bin/activate && \
    cd /opt/antenna_designer && \
    pip install PyNEC --no-index --find-links https://github.com/stevenmburns/python-necpp/releases/expanded_assets/v1.7.4-accel.1 && \
    git submodule update --init pysim && \
    pip install --no-build-isolation -e ./pysim && \
    pip install -e ."
