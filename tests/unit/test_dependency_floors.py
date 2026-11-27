"""Guard the security floor on keras in requirements.txt and pyproject.toml.

keras 3.15.1 restricts unpickling in the IMDB/Reuters/CIFAR dataset loaders
and verifies every intermediary H5 group when loading weights (path
traversal). Releases from 3.13 on need Python 3.11+, so 3.10 cannot reach
3.15.1; it is pinned to 3.12.4 instead, the last release that still installs
on 3.10 and the one that carries every fix available there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

packaging_requirements = pytest.importorskip("packaging.requirements")
Requirement = packaging_requirements.Requirement

ROOT = Path(__file__).resolve().parents[2]

# Highest keras release whose wheel declares Requires-Python >=3.10.
KERAS_LAST_PY310 = "3.12.4"
# Highest keras release that carries the unpickling / H5 path-traversal fixes.
KERAS_PY311_FLOOR = "3.15.1"


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
    assert spec.contains(KERAS_PY311_FLOOR)


# Versions a Python 3.10 install must refuse. Each one is affected by a CVE
# that is fixed at or before 3.12.4, so there is no reason to allow it.
PY310_VULNERABLE = [
    "3.0.0",  # CVE-2025-1550 (unsafe Lambda deserialization)
    "3.7.0",  # CVE-2024-55459 (get_file arbitrary file write)
    "3.8.0",  # CVE-2025-8747 (safe_mode bypass via crafted config)
    "3.11.0",  # CVE-2025-9905 / CVE-2026-1669
    "3.11.3",  # CVE-2025-12060 (H5 path traversal)
    "3.12.0",  # CVE-2025-12058 (load_model arbitrary local file load)
    "3.12.3",  # last release before the 3.10-compatible fix set is complete
]


@pytest.mark.parametrize("source", [_requirements_txt, _pyproject_dependencies])
def test_python_310_floor_refuses_vulnerable_releases(source):
    keras = _keras_for(source(), "3.10")
    assert len(keras) == 1
    spec = keras[0].specifier
    for version in PY310_VULNERABLE:
        assert not spec.contains(version), (
            f"Python 3.10 floor still admits keras {version}; raise it to "
            f">={KERAS_LAST_PY310}"
        )


@pytest.mark.parametrize("source", [_requirements_txt, _pyproject_dependencies])
def test_python_310_floor_is_the_best_installable_release(source):
    keras = _keras_for(source(), "3.10")
    assert len(keras) == 1
    spec = keras[0].specifier
    assert spec.contains(KERAS_LAST_PY310), (
        f"Python 3.10 must be able to install keras {KERAS_LAST_PY310}, "
        "the last release with Requires-Python >=3.10"
    )
    # 3.13 is unreachable on 3.10; the cap keeps pip's resolver from
    # backtracking through releases it can never use.
    assert not spec.contains("3.13.0"), "keras 3.13+ requires Python 3.11"
