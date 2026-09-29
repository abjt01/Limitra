"""Release helper for the publish workflow.

``check`` decides whether the current commit should be released: it should
if ``limitra.__version__`` has not been released yet. ``notes`` prints that
version's section of CHANGELOG.md for the GitHub release.

Run from the repository root::

    python .github/release.py check
    python .github/release.py notes 0.2.1
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PACKAGE = "abjt-limitter"
VERSION_FILE = Path("src/limitra/__init__.py")
CHANGELOG = Path("CHANGELOG.md")


def current_version() -> str:
    """Return ``__version__`` as written in the package."""
    match = re.search(r'^__version__ = "([^"]+)"$', VERSION_FILE.read_text(), re.M)
    if not match:
        sys.exit(f"::error::no __version__ found in {VERSION_FILE}")
    version = match.group(1)
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        sys.exit(f"::error::__version__ {version!r} is not MAJOR.MINOR.PATCH")
    return version


def released_versions() -> set[str]:
    """Return every version already released, from git tags and PyPI.

    Git tags are checked as well as PyPI because PyPI's JSON API sits
    behind a CDN and can lag a fresh upload by a few minutes; the tag is
    pushed straight after the upload and is visible immediately.
    """
    tags = subprocess.run(
        ["git", "tag", "--list", "v*"], capture_output=True, text=True, check=True
    ).stdout.split()
    released = {tag[1:] for tag in tags}
    try:
        url = f"https://pypi.org/pypi/{PACKAGE}/json"
        with urllib.request.urlopen(url, timeout=30) as response:
            released |= set(json.load(response)["releases"])
    except urllib.error.HTTPError as exc:
        if exc.code != 404:  # 404 only means nothing has been published yet
            raise
    return released


def key(version: str) -> tuple[int, ...]:
    """Sort key for a MAJOR.MINOR.PATCH version string."""
    return tuple(int(part) for part in version.split("."))


def check() -> None:
    """Write ``version`` and ``publish`` to ``$GITHUB_OUTPUT``."""
    version = current_version()
    released = {v for v in released_versions() if re.fullmatch(r"\d+\.\d+\.\d+", v)}
    newer = sorted((v for v in released if key(v) > key(version)), key=key)

    if version in released:
        publish = False
        print(f"::notice::{version} is already released; bump __version__ to publish")
    elif newer:
        sys.exit(
            f"::error::__version__ is {version}, but {newer[-1]} is already "
            f"released. Versions must only go up."
        )
    else:
        publish = True
        print(f"::notice::publishing {version}")

    outputs = f"version={version}\npublish={'true' if publish else 'false'}\n"
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(outputs)
    else:
        print(outputs, end="")


def notes(version: str) -> None:
    """Print CHANGELOG.md's section for ``version``, or nothing if absent."""
    text = CHANGELOG.read_text() if CHANGELOG.exists() else ""
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.M | re.S
    )
    if match:
        print(match.group(1).strip())


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "check":
        check()
    elif command == "notes" and len(sys.argv) == 3:
        notes(sys.argv[2])
    else:
        sys.exit("usage: release.py check | release.py notes VERSION")
