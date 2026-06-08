# antenna_designer web UI

A schema-driven web UI: the backend introspects every concrete `Builder`
under `src/antenna_designer/designs/**` and emits its parameter schema
(default + range) so the frontend can generate sliders without
geometry-specific code.

## Dev launch

Two processes:

```bash
# Terminal 1 — FastAPI backend on :8000
source .venv/bin/activate
python -m uvicorn web.server:app --reload --port 8000

# Terminal 2 — Vite dev server on :5173 (proxies /api → :8000)
cd web/frontend
npm install     # first time
npm run dev
```

Open <http://127.0.0.1:5173>.

## Endpoints (Phase 1)

* `GET  /api/builders` — every concrete Builder with its variants + slider schema
* `GET  /api/builder/{name}?variant=<v>` — schema for one builder + variant
* `POST /api/solve` — `{builder, variant, params, engine, pysim_basis?, ground?, far_field?}`
   → `{z_per_feed, wires, currents, far_field?}`

## Tuning slider ranges

Generic ranges come from a name-keyed heuristic in
`src/antenna_designer/web_schema.py`. To override ranges for a specific
design, add a `param_ranges` class attribute alongside `default_params`:

```python
class Builder(AntennaBuilder):
    default_params = MappingProxyType({"length_factor": 0.48, ...})
    param_ranges = {"length_factor": (0.4, 0.55), "gap_z": (0.05, 0.5, 0.005)}
```

Tuples are `(min, max)` or `(min, max, step)`.
