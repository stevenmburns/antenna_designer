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
    # C++ Dependencies
    g++ \
    gfortran \
    swig3.0 \
    git \
    build-essential \
    automake \
    libtool \
&&    apt-get -qq clean

FROM antenna_designer_base AS antenna_designer_base_with_python

# Create virtual environment and install python packages
# Upgrade pip & install testing dependencies
# Note: Package dependencies are in setup.py
RUN \
    /bin/bash -c "python3 -m venv /opt/.venv && \
    source /opt/.venv/bin/activate && \
    pip install --upgrade pip -q && \
    pip install setuptools numpy scipy pytest matplotlib icecream scikit-rf -q"

FROM antenna_designer_base_with_python as antenna_designer

COPY . /opt/antenna_designer

RUN \
    /bin/bash -c "source /opt/.venv/bin/activate && \
    cd /opt && \
    cd antenna_designer && \
    git submodule init && \
    git submodule update --remote && \
    pushd python-necpp && \    
    git submodule init && \
    git submodule update --remote && \
    cd PyNEC && \
    ln -s ../necpp_src . && \
    pushd ../necpp_src && \
    make -f Makefile.git && \
    ./configure --without-lapack && \
    popd && \
    swig3.0 -Wall -v -c++ -python PyNEC.i && \
    python setup.py build && \
    python setup.py install && \
    popd && \
    pip install -e ."
