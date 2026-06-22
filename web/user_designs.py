"""Web-layer glue for user-authored antenna designs.

The discovery + path-loading core lives in ``antenna_designer.user_designs``
(no web dependency, shared with the CLI). This module layers on the bits that
are specific to the running web app: registering each design into ``REGISTRY``
under ``user.<filename>``, scaffolding the folder on first run, and formatting
load errors for the "failed to load" UI panel.
"""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path

from antenna_designer.user_designs import (
    USER_NS,
    default_user_dir,
    iter_design_files,
    load_builder,
    user_design_dirs,
)

from . import adapter
from .examples import REGISTRY

__all__ = [
    "USER_NS",
    "default_user_dir",
    "user_design_dirs",
    "ensure_scaffold",
    "refresh",
]

_ASSETS = Path(__file__).resolve().parent / "user_design_assets"


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
    for stem, path in iter_design_files():
        name = f"{USER_NS}.{stem}"
        try:
            cls = load_builder(path)
            # Smoke-test: construct, then actually build the geometry so a
            # typo / bad shape in build_wires surfaces here (at load, in the UI
            # panel) instead of only when the design is solved.
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
