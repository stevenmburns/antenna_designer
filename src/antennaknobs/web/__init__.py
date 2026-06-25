"""antennaknobs web API server.

A FastAPI app (``web.server:app``) plus its antenna-example registry
(``web.examples``) and the adapter that bridges antennaknobs's Builders to the
JSON/WebSocket contract the frontend consumes. The React frontend under
``web/frontend`` is a separate build and is not part of this package.

This ``__init__`` exists so ``web`` is a real importable package when
antennaknobs is installed from a wheel (not just from a source checkout with
the repo root on ``sys.path``).
"""
