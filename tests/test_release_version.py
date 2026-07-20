"""Tests for the GitHub release-version helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "set_release_version.py"
)
SPEC = importlib.util.spec_from_file_location("set_release_version", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SET_RELEASE_VERSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SET_RELEASE_VERSION)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("v0.1.9", "0.1.9"), ("1.2.3", "1.2.3"), ("v10.20.30", "10.20.30")],
)
def test_version_from_tag(tag, expected):
    assert SET_RELEASE_VERSION.version_from_tag(tag) == expected


@pytest.mark.parametrize("tag", ["v0.1", "release-0.1.9", "v01.2.3", "v1.2.3.4"])
def test_version_from_tag_rejects_invalid_tags(tag):
    with pytest.raises(ValueError, match="Expected vX.Y.Z"):
        SET_RELEASE_VERSION.version_from_tag(tag)


def test_update_project_version_changes_only_project_table(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'name = "example"\n'
        'version = "0.1.8"\n\n'
        "[tool.example]\n"
        'version = "keep-this-value"\n',
        encoding="utf-8",
    )

    previous = SET_RELEASE_VERSION.update_project_version(pyproject, "0.1.9")
    updated = pyproject.read_text(encoding="utf-8")

    assert previous == "0.1.8"
    assert 'version = "0.1.9"' in updated
    assert 'version = "keep-this-value"' in updated
