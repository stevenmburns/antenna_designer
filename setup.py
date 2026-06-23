from setuptools import find_packages, setup

# antennaknobs uses a src-layout (src/antennaknobs, with the antennaknobs.engines
# subpackage); the web API server lives at the repo root as the `web` package
# (with its web.examples antenna registry). Discover both and map each top-level
# package to its directory — a plain packages=["antennaknobs"] would silently
# drop antennaknobs.engines and all of web from a built wheel (editable installs
# mask this because they put the source roots on sys.path wholesale).
#
# web/user_design_assets/ is not a package (no __init__); it holds TEMPLATE.py +
# CLAUDE.md that web/user_designs.py copies into the user folder on first run, so
# it ships as package data of the `web` package.
packages = find_packages(where="src") + find_packages(include=["web", "web.*"])

setup(
    packages=packages,
    package_dir={"antennaknobs": "src/antennaknobs", "web": "web"},
    package_data={"web": ["user_design_assets/*"]},
)
