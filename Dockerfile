# Multi-stage build for the antennaknobs web workbench (the live simulator).
#
# Stage 1 builds the React/Vite SPA to src/antennaknobs/web/static.
# Stage 2 is a slim Python runtime that installs the package + its C++ engine
# wheels (momwire, pynec-accel) from TestPyPI and serves everything from one
# uvicorn process (API + the /ws live-solve WebSocket + the static SPA).
#
# See docs/deploy.md for the build/run/deploy runbook.

# ---- Stage 1: build the frontend -------------------------------------------
# Vite 8 needs Node >=22.12 (or >=20.19); node:22 satisfies it.
FROM node:22-bookworm-slim AS frontend

# Mirror the repo path so vite's outDir ("../static") lands at
# /app/src/antennaknobs/web/static — exactly where server.py looks for it.
WORKDIR /app/src/antennaknobs/web/frontend

# Install deps first (cached unless the lockfile changes).
COPY src/antennaknobs/web/frontend/package.json src/antennaknobs/web/frontend/package-lock.json ./
RUN npm ci

# Then the sources, and build.
COPY src/antennaknobs/web/frontend/ ./
RUN npm run build   # writes /app/src/antennaknobs/web/static


# ---- Stage 2: python runtime -----------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# libgomp1: the OpenMP runtime the C++ accelerators link against. Both momwire
# (>=0.2.2) and pynec-accel (>=1.7.4.post1) de-vendor libgomp and link THIS
# system copy, so they share one OpenMP runtime — no private-vendored static-TLS
# clash, so no GLIBC_TUNABLES workaround is needed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Package metadata + sources (an editable install keeps server.py's
# _FRONTEND_DIR = <module>/static pointing at the bundle we copy in below).
COPY pyproject.toml setup.py README.md ./
COPY src/ ./src/

# The built SPA from stage 1.
COPY --from=frontend /app/src/antennaknobs/web/static ./src/antennaknobs/web/static

# The C++ engine wheels live on TestPyPI; their scientific deps (numpy/scipy)
# come from real PyPI. Install those FIRST so the editable install below sees
# the momwire requirement already satisfied — and so the second install can use
# real PyPI ONLY. (TestPyPI must NOT be an index for the `[web]` deps: it hosts a
# stray, unbuildable `fastapi` sdist that shadows the real wheel and breaks the
# build.)
RUN pip install --upgrade pip \
 && pip install \
      --index-url https://test.pypi.org/simple/ \
      --extra-index-url https://pypi.org/simple/ \
      "momwire>=0.2.2" "pynec-accel>=1.7.4.post1" \
 && pip install -e ".[web]"

EXPOSE 8000

# One worker: solves are CPU-bound and run in a threadpool, so extra uvicorn
# workers would only contend for the same cores. --host 0.0.0.0 to accept the
# proxy's traffic inside the container.
CMD ["sh", "-c", "uvicorn antennaknobs.web.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
