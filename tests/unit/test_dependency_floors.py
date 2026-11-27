"""Guard the security floor on keras in requirements.txt and pyproject.toml.

keras 3.15.1 restricts unpickling in the IMDB/Reuters/CIFAR dataset loaders
and verifies every intermediary H5 group when loading weights (path
traversal). Releases from 3.13 on need Python 3.11+, so every keras release
that carries the later fixes (CVE-2026-1462, CVE-2026-11816, CVE-2026-12570)
is out of reach on Python 3.10. The project therefore requires Python 3.11+
and has a single, unconditional keras floor.

These tests pin three things: the floor is at least 3.15.1 for every Python
version the project supports; the project declares that it needs 3.11+; and
no Python version gets a second, weaker keras requirement.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

packaging_requirements = pytest.importorskip("packaging.requirements")
Requirement = packaging_requirements.Requirement
SpecifierSet = pytest.importorskip("packaging.specifiers").SpecifierSet

ROOT = Path(__file__).resolve().parents[2]

# Lowest keras release that carries the unpickling / H5 path-traversal fixes
# and the three later fixes named in the module docstring.
KERAS_FLOOR = "3.15.1"
# Lowest Python the project supports. keras 3.13+ needs 3.11.
MIN_PYTHON = "3.11"


def _requirements_txt() -> list[str]:
    lines = []
    for raw in (ROOT / "requirements.txt").read_text().splitlines():
        line = re.sub(r"\s+#.*$", "", raw).strip()
        if line and not line.startswith(("#", "-")):
            lines.append(line)
    return lines


def _pyproject_dependencies() -> list[str]:
    text = (ROOT / "pyproject.toml").read_text()
    block = re.search(r"^dependencies = \[(.*?)^\]", text, re.S | re.M)
    assert block, "pyproject.toml has no [project] dependencies block"
    return re.findall(r'^\s*"([^"]+)"', block.group(1), re.M)


def _keras_for(lines: list[str], python: str) -> list[Requirement]:
    env = {"python_version": python, "python_full_version": f"{python}.0"}
    reqs = [Requirement(line) for line in lines]
    return [
        r
        for r in reqs
        if r.name.lower() == "keras" and (r.marker is None or r.marker.evaluate(env))
    ]


@pytest.mark.parametrize("source", [_requirements_txt, _pyproject_dependencies])
@pytest.mark.parametrize("python", ["3.11", "3.12", "3.13", "3.14"])
def test_keras_floor_excludes_unpatched_releases(source, python):
    keras = _keras_for(source(), python)
    assert len(keras) == 1, f"expected one keras requirement for Python {python}"
    spec = keras[0].specifier
    for version in UNPATCHED:
        assert not spec.contains(version), f"keras {version} is affected by a CVE fixed at or before {KERAS_FLOOR}"
    assert spec.contains(KERAS_FLOOR)


# Releases an install must refuse, each tied to the CVE fixed in it or later.
UNPATCHED = [
    "3.0.0",  # CVE-2025-1550 (unsafe Lambda deserialization)
    "3.7.0",  # CVE-2024-55459 (get_file arbitrary file write)
    "3.8.0",  # CVE-2025-8747 (safe_mode bypass via crafted config)
    "3.11.0",  # CVE-2025-9905 / CVE-2026-1669
    "3.11.3",  # CVE-2025-12060 (H5 path traversal)
    "3.12.0",  # CVE-2025-12058 (load_model arbitrary local file load)
    "3.12.4",  # CVE-2026-1462 (arbitrary code execution), fixed in 3.13.2
    "3.13.2",  # CVE-2026-11816 (archive path traversal), fixed in 3.14.0
    "3.14.0",  # CVE-2026-12570 (HDF5 shape bomb), fixed in 3.15.0
    "3.15.0",  # lacks the unpickling / H5 group-traversal fixes of 3.15.1
]


def _requires_python() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^requires-python\s*=\s*"([^"]+)"', text, re.M)
    assert match, "pyproject.toml does not declare requires-python"
    return match.group(1)


def test_project_declares_python_311_as_its_floor():
    """The single keras floor is only honest if 3.10 is not advertised.

    keras 3.13+ needs Python 3.11, so a project that still accepts 3.10 would
    hand those installs an unfixable keras (CVE-2026-1462, CVE-2026-11816,
    CVE-2026-12570). requires-python is what pip enforces, so it must agree
    with the markers on the keras requirement.
    """
    spec = SpecifierSet(_requires_python())
    assert spec.contains(MIN_PYTHON), (
        f"requires-python is {_requires_python()!r}; it must admit {MIN_PYTHON}"
    )
    for older in ("3.10", "3.9"):
        assert not spec.contains(older), (
            f"requires-python {_requires_python()!r} still admits Python {older}, "
            f"where keras cannot reach {KERAS_FLOOR}"
        )


@pytest.mark.parametrize("source", [_requirements_txt, _pyproject_dependencies])
def test_keras_floor_is_unconditional(source):
    """The keras floor carries no environment marker.

    requires-python is what keeps Python 3.10 out, and it is tested above.
    The keras requirement is the line of defence behind it, so it must not
    have a marker of its own: a `python_version >= '3.11'` marker would make
    an install that slips past requires-python succeed *silently with no
    keras at all*, whereas an unmarked floor makes the resolver fail loudly.
    """
    keras = [r for r in map(Requirement, source()) if r.name.lower() == "keras"]
    assert len(keras) == 1, f"expected exactly one keras requirement, got {keras}"
    assert keras[0].marker is None, (
        f"keras requirement is gated by a marker ({keras[0].marker!r}); it must "
        "apply on every Python so an unsupported one fails loudly"
    )


@pytest.mark.parametrize("source", [_requirements_txt, _pyproject_dependencies])
def test_keras_requirement_has_no_upper_cap(source):
    """No `<3.x` cap: a cap would hold installs back from future security fixes."""
    keras = _keras_for(source(), "3.12")
    assert len(keras) == 1
    capped = [s for s in keras[0].specifier if s.operator in ("<", "<=", "!=")]
    assert not capped, f"keras is capped: {keras[0].specifier}"
