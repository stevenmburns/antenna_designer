from setuptools import find_packages, setup

# antennaknobs uses a src-layout (src/antennaknobs). The web API server lives at
# src/antennaknobs/web (the `antennaknobs.web` subpackage, with its
# antennaknobs.web.examples antenna registry) — it used to be a top-level `web`
# package at the repo root, but that shipped a dangerously generic `web` module
# into site-packages, so it now lives under the namespace. find_packages(where=
# "src") discovers antennaknobs and every subpackage (engines, web, web.examples)
# in one sweep; a plain packages=["antennaknobs"] would silently drop them from a
# built wheel (editable installs mask this because they put the source root on
# sys.path wholesale).
#
# antennaknobs/web/user_design_assets/ is not a package (no __init__); it holds
# TEMPLATE.py + CLAUDE.md that antennaknobs/web/user_designs.py copies into the
# user folder on first run, so it ships as package data of antennaknobs.web.
#
# antennaknobs/web/static/ is the built React frontend (vite `npm run build`
# output). It is gitignored and produced at publish time; shipping it as package
# data is what lets a wheel install serve the browser UI from server.py with no
# Node. A source/editable install without a build simply has no web/static and
# runs API-only (the dev workflow uses the Vite dev server instead).
setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"antennaknobs.web": ["user_design_assets/*", "static/**/*"]},
)
