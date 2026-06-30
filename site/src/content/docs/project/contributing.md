---
title: Contributing
description: How to add a design, run the tests, and submit changes.
---

antennaknobs is open source on
[GitHub](https://github.com/stevenmburns/antennaknobs). Contributions — in the
form of new feature requests (issues) or new designs and fixes (pull requests) —
are welcome.

## Adding a design

A design is a single `Builder` subclass (see [The model](/concepts/model/)). The
clearest path is to copy a design from the [catalog](/reference/catalog/) whose
shape is closest to yours and adapt it — that's exactly what the built-ins are
for. If your geometry is path-shaped (a loop, an arm, a zig-zag), the
[`Drone`](/concepts/authoring/#the-drone) is often the clearest way to express it.

Keep to the house conventions: angles in **degrees** (`_deg` suffix), segment
counts derived from a reference length, and a `ui_params` block so the web knobs
get sensible ranges.

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

The suite covers the solver, the design catalog, and the web server; geometry is
checked to machine precision, so new designs should come with a structural test.

## Submitting changes

Work on a branch, keep commits tidy (the repo uses rebase-merge, so each commit
lands on `main` verbatim), and open a PR against `main`.

<!-- TODO: link the repo's actual CONTRIBUTING / dev-setup docs and CI specifics
     once consolidated. -->
