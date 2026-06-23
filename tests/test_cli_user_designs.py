"""CLI resolution of user-authored designs.

The web app registers user files under ``user.<name>`` in its REGISTRY; the
CLI resolves the same ``user.<name>`` spec straight from the folder via
``antennaknobs.user_designs`` (no web dependency). These cover the CLI /
core path, separate from the web-registry behaviour in test_user_designs.py.
"""

import pytest

import antennaknobs as ant
from antennaknobs import user_designs
from antennaknobs.cli import resolve_class

VALID = """
from types import MappingProxyType
from antennaknobs import AntennaBuilder

class Builder(AntennaBuilder):
    label = "CLI test dipole"
    default_params = MappingProxyType({"freq": 14.0, "half_length": 5.0})

    def build_wires(self):
        h = self.half_length
        n = self.nominal_nsegs
        return [
            ((0.0, -h, 0.0), (0.0, -0.01, 0.0), n, None),
            ((0.0, 0.01, 0.0), (0.0, h, 0.0), n, None),
            ((0.0, -0.01, 0.0), (0.0, 0.01, 0.0), 1, 1 + 0j),
        ]
"""

SYNTAX_ERROR = "class Builder(  # unterminated\n"


@pytest.fixture
def userdir(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTENNAKNOBS_USER_DIR", str(tmp_path))
    return tmp_path


def test_resolve_user_design_by_explicit_name(userdir):
    (userdir / "cli_dipole.py").write_text(VALID)
    cls = resolve_class("user.cli_dipole")
    assert cls is not None
    assert cls.__name__ == "Builder"


def test_bare_name_does_not_resolve_user_design(userdir):
    """A user file is reachable only through its `user.` namespace — a bare
    name must not resolve to it (that's what keeps user files from shadowing
    built-ins)."""
    (userdir / "cli_dipole.py").write_text(VALID)
    assert resolve_class("cli_dipole") is None


def test_unknown_user_design_resolves_to_none(userdir):
    assert resolve_class("user.does_not_exist") is None


def test_cli_draw_runs_a_user_design(userdir):
    (userdir / "cli_dipole.py").write_text(VALID)
    ant.cli("draw --builder user.cli_dipole --fn /dev/null".split())


def test_load_error_propagates_with_real_cause(userdir):
    """A broken user file surfaces the actual error (so the author can fix it),
    not a generic 'unknown builder'."""
    (userdir / "broken.py").write_text(SYNTAX_ERROR)
    with pytest.raises(SyntaxError):
        resolve_class("user.broken")


def test_resolve_user_design_helper(userdir):
    (userdir / "cli_dipole.py").write_text(VALID)
    assert user_designs.resolve_user_design("cli_dipole").__name__ == "Builder"
    assert user_designs.resolve_user_design("missing") is None


def test_cli_list_includes_builtin_and_user(userdir, capsys):
    (userdir / "cli_dipole.py").write_text(VALID)
    ant.cli(["list"])
    out = capsys.readouterr().out
    assert "beams.moxon" in out
    assert "user.cli_dipole" in out


def test_cli_list_filter(userdir, capsys):
    ant.cli(["list", "moxon"])
    out = capsys.readouterr().out
    assert "beams.moxon" in out
    assert "arrays.moxonarray" in out
    assert "dipoles.invvee" not in out


def test_cli_list_user_only(userdir, capsys):
    (userdir / "cli_dipole.py").write_text(VALID)
    ant.cli(["list", "--user-only"])
    out = capsys.readouterr().out
    assert "user.cli_dipole" in out
    assert "beams.moxon" not in out


def test_cli_list_builtin_only(userdir, capsys):
    (userdir / "cli_dipole.py").write_text(VALID)
    ant.cli(["list", "--builtin-only"])
    out = capsys.readouterr().out
    assert "beams.moxon" in out
    assert "user." not in out


def test_cli_list_no_match(userdir, capsys):
    ant.cli(["list", "definitely_not_a_design_xyz"])
    out = capsys.readouterr().out
    assert "no designs" in out
