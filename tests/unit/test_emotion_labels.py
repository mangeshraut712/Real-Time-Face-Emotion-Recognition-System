"""Unit tests for FER2013 one-hot label encoding.

Loaded from the source file directly so this test does not pull in OpenCV
via src/__init__.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_LABELS = Path(__file__).resolve().parents[2] / "src" / "utils" / "labels.py"
_spec = importlib.util.spec_from_file_location("emotion_labels", _LABELS)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
encode_emotion_labels = _mod.encode_emotion_labels
NUM_EMOTIONS = _mod.NUM_EMOTIONS


def test_encode_emotion_labels_keeps_all_classes_when_some_are_missing() -> None:
    encoded = encode_emotion_labels(np.array([3, 3, 4]))
    assert encoded.shape == (3, NUM_EMOTIONS)
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(encoded[0], np.eye(NUM_EMOTIONS)[3])
    np.testing.assert_array_equal(encoded[2], np.eye(NUM_EMOTIONS)[4])


def test_encode_emotion_labels_rejects_out_of_range_ids() -> None:
    with pytest.raises(ValueError, match="0..6"):
        encode_emotion_labels(np.array([0, 7]))


def test_encode_emotion_labels_empty_input_has_seven_columns() -> None:
    encoded = encode_emotion_labels(np.array([], dtype=int))
    assert encoded.shape == (0, NUM_EMOTIONS)
