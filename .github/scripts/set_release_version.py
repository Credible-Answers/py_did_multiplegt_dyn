"""Set the package version from a GitHub release tag."""

from __future__ import annotations

import re
import sys
from pathlib import Path


TAG_PATTERN = re.compile(
    r"^v?(?P<version>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))$"
)
VERSION_LINE_PATTERN = re.compile(
    r'^(?P<indent>\s*)version\s*=\s*"(?P<version>[^"]+)"(?P<suffix>\s*(?:#.*)?)$'
)
TABLE_PATTERN = re.compile(r"^\s*\[([^]]+)]\s*(?:#.*)?$")


def version_from_tag(tag: str) -> str:
    """Return ``X.Y.Z`` from a release tag formatted as ``vX.Y.Z`` or ``X.Y.Z``."""
    match = TAG_PATTERN.fullmatch(tag.strip())
    if match is None:
        raise ValueError(
            f"Invalid release tag {tag!r}. Expected vX.Y.Z, for example v0.1.9."
        )
    return match.group("version")


def update_project_version(pyproject_path: Path, version: str) -> str:
    """Update only the ``version`` field in the TOML ``[project]`` table."""
    text = pyproject_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    in_project_table = False
    previous_version = None
    replacements = 0

    for index, line in enumerate(lines):
        content = line.rstrip("\r\n")
        newline = line[len(content):]

        table_match = TABLE_PATTERN.fullmatch(content)
        if table_match is not None:
            in_project_table = table_match.group(1).strip() == "project"
            continue

        if not in_project_table:
            continue

        version_match = VERSION_LINE_PATTERN.fullmatch(content)
        if version_match is None:
            continue

        previous_version = version_match.group("version")
        lines[index] = (
            f'{version_match.group("indent")}version = "{version}"'
            f'{version_match.group("suffix")}{newline}'
        )
        replacements += 1

    if replacements != 1 or previous_version is None:
        raise ValueError(
            "Expected exactly one version field in the [project] table of "
            f"{pyproject_path}; found {replacements}."
        )

    with pyproject_path.open("w", encoding="utf-8", newline="") as output:
        output.write("".join(lines))
    return previous_version


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "Usage: set_release_version.py RELEASE_TAG [PYPROJECT_PATH]"
        )

    tag = sys.argv[1]
    pyproject_path = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("pyproject.toml")
    version = version_from_tag(tag)
    previous_version = update_project_version(pyproject_path, version)
    print(
        f"Package version set from release tag: {previous_version} -> {version} "
        f"({tag})"
    )


if __name__ == "__main__":
    main()
