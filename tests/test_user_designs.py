"""User-authored design discovery, live reload, error surfacing, scaffolding.

See web/user_designs.py — local users drop a Builder file in their user dir
and it registers under `user.<filename>` with no restart.
"""

import pytest

import web.examples  # noqa: F401 — bootstraps the adapter + REGISTRY
from web import user_designs
from web.examples import REGISTRY

VALID = """
from types import MappingProxyType
from antenna_designer import AntennaBuilder

class Builder(AntennaBuilder):
    label = "Test dipole"
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

BROKEN_BUILD = """
from types import MappingProxyType
from antenna_designer import AntennaBuilder

class Builder(AntennaBuilder):
    default_params = MappingProxyType({"freq": 14.0})

    def build_wires(self):
        return undefined_name  # NameError when geometry is built
"""

NO_BUILDER = "x = 1\n"


@pytest.fixture
def userdir(tmp_path, monkeypatch):
    """A clean temp user-design dir; strips any user.* from the shared
    REGISTRY before and after so tests don't leak into each other."""
    monkeypatch.setenv("ANTENNA_DESIGNER_USER_DIR", str(tmp_path))

    def _clear():
        for k in [k for k in REGISTRY if k.startswith("user.")]:
            del REGISTRY[k]

    _clear()
    yield tmp_path
    _clear()


def test_valid_design_registers(userdir):
    (userdir / "my_dipole.py").write_text(VALID)
    errors = user_designs.refresh()
    assert errors == []
    assert "user.my_dipole" in REGISTRY
    assert REGISTRY["user.my_dipole"].name == "user.my_dipole"


def test_broken_geometry_reports_error_without_crashing(userdir):
    (userdir / "oops.py").write_text(BROKEN_BUILD)
    errors = user_designs.refresh()
    assert "user.oops" not in REGISTRY
    assert len(errors) == 1
    assert errors[0]["name"] == "user.oops"
    assert "NameError" in errors[0]["message"]


def test_missing_builder_reports_error(userdir):
    (userdir / "nobuilder.py").write_text(NO_BUILDER)
    errors = user_designs.refresh()
    assert "user.nobuilder" not in REGISTRY
    assert any("Builder" in e["message"] for e in errors)


def test_one_bad_design_does_not_block_a_good_one(userdir):
    (userdir / "good.py").write_text(VALID)
    (userdir / "bad.py").write_text(BROKEN_BUILD)
    errors = user_designs.refresh()
    assert "user.good" in REGISTRY
    assert {e["name"] for e in errors} == {"user.bad"}


def test_reload_picks_up_edits(userdir):
    f = userdir / "d.py"
    f.write_text(VALID)
    assert user_designs.refresh() == []
    assert "user.d" in REGISTRY

    f.write_text(BROKEN_BUILD)  # break it
    errors = user_designs.refresh()
    assert "user.d" not in REGISTRY
    assert errors and errors[0]["name"] == "user.d"

    f.write_text(VALID)  # fix it
    assert user_designs.refresh() == []
    assert "user.d" in REGISTRY


def test_template_file_is_skipped(userdir):
    (userdir / "TEMPLATE.py").write_text(VALID)
    user_designs.refresh()
    assert "user.TEMPLATE" not in REGISTRY


def test_scaffold_creates_assets(tmp_path, monkeypatch):
    target = tmp_path / "designs"
    monkeypatch.setenv("ANTENNA_DESIGNER_USER_DIR", str(target))
    user_designs.ensure_scaffold()
    assert (target / "TEMPLATE.py").is_file()
    assert (target / "CLAUDE.md").is_file()

    # The shipped template must itself be a loadable design (copied under a
    # non-TEMPLATE name, since TEMPLATE.py is skipped by discovery).
    (target / "example_from_template.py").write_text(
        (target / "TEMPLATE.py").read_text()
    )
    errors = user_designs.refresh()
    try:
        assert "user.example_from_template" in REGISTRY
        assert not any(e["name"] == "user.example_from_template" for e in errors)
    finally:
        for k in [k for k in REGISTRY if k.startswith("user.")]:
            del REGISTRY[k]
