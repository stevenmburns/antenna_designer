"""Discovery and live-reload of user-authored antenna designs.

For a local install, a ham (or Claude Code on their behalf) drops a ``.py``
file defining a ``Builder`` into their user design folder; it shows up in the
web UI under the ``user`` family with no server restart. The authoring
contract lives in ``user_design_assets/TEMPLATE.py`` and ``CLAUDE.md``, which
are copied into the user folder on first run.

User designs live *outside* the installed package (so a ``pip`` upgrade never
clobbers them) and are loaded by file path, then registered under
``user.<filename>`` — a namespace that can never collide with or shadow a
built-in family design.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import traceback
from pathlib import Path

from . import adapter
from .examples import REGISTRY

USER_NS = "user"
_ASSETS = Path(__file__).resolve().parent / "user_design_assets"
_MODULE_PREFIX = "antenna_designer._user_designs"


def default_user_dir() -> Path:
    """The primary user-design folder: ``$ANTENNA_DESIGNER_USER_DIR`` if set,
    else ``~/.antenna_designer/designs``."""
    env = os.environ.get("ANTENNA_DESIGNER_USER_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".antenna_designer" / "designs"


def user_design_dirs() -> list[Path]:
    """Folders scanned for user designs, in priority order: the default (or
    the env override) plus ``./antenna_designs`` if it exists in the cwd."""
    dirs = [default_user_dir()]
    seen = {d.resolve() for d in dirs if d.exists()}
    local = Path.cwd() / "antenna_designs"
    if local.is_dir() and local.resolve() not in seen:
        dirs.append(local)
    return dirs


def ensure_scaffold() -> None:
    """Create the default user folder with a ``TEMPLATE.py`` + ``CLAUDE.md``
    the first time, so there's something to copy and Claude Code has context.
    Idempotent and best-effort — never raises into startup."""
    d = default_user_dir()
    if d.exists():
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        for asset in ("TEMPLATE.py", "CLAUDE.md"):
            src = _ASSETS / asset
            if src.is_file():
                shutil.copyfile(src, d / asset)
    except OSError as exc:  # read-only home, permissions, … — log and move on
        print(f"[user_designs] could not scaffold {d}: {exc!r}")


def _load_builder(path: Path):
    """Import a single user file by path and return its ``Builder`` class.
    Re-executes the file on every call, so edits are picked up live."""
    modname = f"{_MODULE_PREFIX}.{path.stem}"
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {path.name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(modname, None)
        raise
    cls = getattr(mod, "Builder", None)
    if cls is None:
        raise AttributeError(
            "no `Builder` class found — define `class Builder(AntennaBuilder): ...`"
        )
    return cls


def _format_error(path: Path, exc: Exception) -> str:
    """A short, user-facing message pointing at the line in *their* file."""
    tb = traceback.extract_tb(exc.__traceback__)
    here = [fr for fr in tb if fr.filename == str(path)]
    where = f" (line {here[-1].lineno})" if here else ""
    return f"{type(exc).__name__}: {exc}{where}"


def refresh() -> list[dict]:
    """Reload every user design into ``REGISTRY`` under ``user.<filename>``,
    replacing any previously-loaded user designs.

    Returns a list of ``{"name", "file", "message"}`` for files that failed to
    load — surfaced in the UI so the author (or Claude) can fix them. A broken
    file never takes down the rest.
    """
    for key in [k for k in REGISTRY if k.startswith(f"{USER_NS}.")]:
        del REGISTRY[key]

    errors: list[dict] = []
    seen: set[str] = set()
    for d in user_design_dirs():
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.py")):
            stem = path.stem
            if stem.startswith("_") or stem == "TEMPLATE":
                continue
            name = f"{USER_NS}.{stem}"
            if name in seen:
                continue  # first folder in priority order wins
            seen.add(name)
            try:
                cls = _load_builder(path)
                # Smoke-test: construct, then actually build the geometry so a
                # typo / bad shape in build_wires surfaces here (at load, in
                # the UI panel) instead of only when the design is solved.
                cls().build_wires()
                REGISTRY[name] = adapter._make_example(name, cls)
            except Exception as exc:  # noqa: BLE001 — surface, don't crash
                errors.append(
                    {
                        "name": name,
                        "file": str(path),
                        "message": _format_error(path, exc),
                    }
                )
    return errors
