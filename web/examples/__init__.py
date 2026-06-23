"""Antenna example registry.

Each antenna geometry exposed by the web UI is defined in its own module
under this package and registered via `register(EXAMPLE)`. The dispatchers
in `web.server` and `web.pynec_backend` look up the example by its `name`
field, so removing an antenna is a two-line change: delete the module file
and the matching `from . import <name>` line below.

The registry is intentionally web-layer (not solver-core) — each example
parses the request dict, calls shared helpers in `web.server` /
`web.pynec_backend`, and produces the JSON-shaped response the frontend
consumes. The solver package `momwire/` stays free of UI concerns.
"""

from __future__ import annotations

from ._base import AntennaExample, ParamSpec

REGISTRY: dict[str, AntennaExample] = {}


def register(example: AntennaExample) -> AntennaExample:
    if example.name in REGISTRY:
        raise ValueError(f"duplicate antenna example: {example.name}")
    REGISTRY[example.name] = example
    return example


# Examples are auto-generated from antenna_designer's designs/ package via
# the adapter, which derives a ParamSpec schema from each Builder's
# default_params and wraps MomwireEngine + PyNECEngine in the SolveFn /
# SweepFn contract above. See web/adapter.py for the bridge and what each
# Builder can opt into via its `ui_params` dict.
from .. import adapter  # noqa: E402

adapter.register_all()

__all__ = ["AntennaExample", "ParamSpec", "REGISTRY", "register"]
