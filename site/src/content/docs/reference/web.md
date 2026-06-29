---
title: Web workbench
description: The browser-based simulator — running it, and how it's served.
---

The web workbench is the live, no-install face of antennaknobs: a panel of knobs
per design, with the radiation pattern, SWR, and impedance re-solving as you drag.

## Run it locally

```bash
pip install "antennaknobs[web]"
uvicorn antennaknobs.web.server:app      # http://127.0.0.1:8000
```

The `[web]` extra pulls in `uvicorn[standard]`, which provides the WebSocket
support the live-solve channel (`/ws`) needs — plain `uvicorn` fails that
handshake.

## The hosted instance

A hosted simulator is running at
**[app.antennaknobs.dev](https://app.antennaknobs.dev/)** (a single
FastAPI process serving the API, the `/ws` live-solve channel, and the built
React SPA). It's deployed as a container on Fly.io; the repo's `docs/deploy.md`
is the runbook.

## How a knob turn works

A knob change sends one message over the `/ws` WebSocket; the server re-solves in
a worker thread and sends the result back. Perceived latency is dominated by the
**solve time** (free-space dipole-class solves are tens of milliseconds), not the
network — so a regional server feels responsive for live tuning.

<!-- TODO: document the FastAPI routes (/sweep, /pattern, /converge, /export_nec,
     /geometry, /examples, /ws) and the ParamSpec / _auto_paramspec knob model. -->
