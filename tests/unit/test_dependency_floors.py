"""Guard the security floor on keras in requirements.txt and pyproject.toml.

keras 3.15.1 restricts unpickling in the IMDB/Reuters/CIFAR dataset loaders
and verifies every intermediary H5 group when loading weights (path
traversal). Releases from 3.13 on need Python 3.11+, so 3.10 keeps the old
floor behind an environment marker.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

packaging_requirements = pytest.importorskip("packaging.requirements")
Requirement = packaging_requirements.Requirement

ROOT = Path(__file__).resolve().parents[2]


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
@pytest.mark.parametrize("python", ["3.11", "3.12", "3.13"])
def test_keras_floor_excludes_unpatched_releases(source, python):
    keras = _keras_for(source(), python)
    assert len(keras) == 1, f"expected one keras requirement for Python {python}"
    spec = keras[0].specifier
    assert not spec.contains("3.15.0"), "keras 3.15.0 lacks the unpickling/H5 fixes"
    assert spec.contains("3.15.1")


@pytest.mark.parametrize("source", [_requirements_txt, _pyproject_dependencies])
def test_python_310_still_resolves(source):
    # requires-python is >=3.10, and no patched keras supports 3.10.
    keras = _keras_for(source(), "3.10")
    assert len(keras) == 1
    assert keras[0].specifier.contains("3.12.1")
