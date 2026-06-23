"""Built-in antenna design catalog.

Each design is a Builder module under a family subpackage
(``antennaknobs.designs.<family>.<name>``) and is addressed by the CLI/web as
``<family>.<name>`` (e.g. ``dipoles.invvee``). This is a regular package (not an
implicit namespace package) so the whole catalog ships in the built wheel;
user-authored designs live in the separate ``user`` namespace, not here.
"""
