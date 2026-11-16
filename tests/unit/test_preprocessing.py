"""Unit tests for preprocessing utilities.

Loaded from the source file directly so this test does not pull in TensorFlow/Keras
via src/__init__.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if "src.utils.labels" not in sys.modules:
    _labels_path = _ROOT / "src" / "utils" / "labels.py"
    _labels_spec = importlib.util.spec_from_file_location("src.utils.labels", _labels_path)
    assert _labels_spec is not None and _labels_spec.loader is not None
    _labels_mod = importlib.util.module_from_spec(_labels_spec)
    sys.modules["src.utils.labels"] = _labels_mod
    _labels_spec.loader.exec_module(_labels_mod)

_prep_path = _ROOT / "src" / "utils" / "preprocessing.py"
_prep_spec = importlib.util.spec_from_file_location("src.utils.preprocessing", _prep_path)
assert _prep_spec is not None and _prep_spec.loader is not None
_prep_mod = importlib.util.module_from_spec(_prep_spec)
sys.modules["src.utils.preprocessing"] = _prep_mod
_prep_spec.loader.exec_module(_prep_mod)

normalize_face = _prep_mod.normalize_face
preprocess_input = _prep_mod.preprocess_input

import numpy as np
import pytest


def test_preprocess_input_v2():
    x = np.array([0.0, 127.5, 255.0], dtype=np.float32)
    out = preprocess_input(x, v2=True)
    assert pytest.approx(out[0]) == -1.0
    assert pytest.approx(out[1], abs=0.01) == 0.0
    assert pytest.approx(out[2]) == 1.0


def test_preprocess_input_v1():
    x = np.array([0.0, 127.5, 255.0], dtype=np.float32)
    out = preprocess_input(x, v2=False)
    assert pytest.approx(out[0]) == 0.0
    assert pytest.approx(out[1], abs=0.01) == 0.5
    assert pytest.approx(out[2]) == 1.0


def test_normalize_face_float_zero_to_one():
    # Issue #31: Float crop in [0, 1] must not equalize to all zeros (-1.0 output)
    face_float = np.linspace(0.1, 0.9, 100 * 100, dtype=np.float32).reshape(100, 100)
    out = normalize_face(face_float, target_size=(64, 64))
    assert out.shape == (64, 64, 1)
    assert not np.all(out == -1.0)
    assert out.min() < out.max()


def test_normalize_face_float_minus_one_to_one():
    face_float = np.linspace(-0.8, 0.8, 100 * 100, dtype=np.float32).reshape(100, 100)
    out = normalize_face(face_float, target_size=(64, 64))
    assert out.shape == (64, 64, 1)
    assert not np.all(out == -1.0)
    assert out.min() < out.max()


def test_normalize_face_uint8():
    face_uint8 = np.linspace(25, 230, 100 * 100, dtype=np.uint8).reshape(100, 100)
    out = normalize_face(face_uint8, target_size=(48, 48))
    assert out.shape == (48, 48, 1)
    assert not np.all(out == -1.0)
    assert out.min() < out.max()


def test_normalize_face_bgr():
    face_bgr = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    out = normalize_face(face_bgr, target_size=(64, 64))
    assert out.shape == (64, 64, 1)


def test_normalize_face_single_channel_dim():
    face_single_ch = np.random.randint(0, 255, (80, 80, 1), dtype=np.uint8)
    out = normalize_face(face_single_ch, target_size=(64, 64))
    assert out.shape == (64, 64, 1)
