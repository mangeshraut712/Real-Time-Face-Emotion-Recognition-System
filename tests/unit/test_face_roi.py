"""Unit tests for Face.extract_roi padding and bounds clamping."""

from __future__ import annotations

import numpy as np

from src.core.face_detector import Face


def _grid(h: int, w: int) -> np.ndarray:
    """Image with unique pixel values so crops can be compared exactly."""
    return np.arange(h * w, dtype=np.uint32).reshape(h, w)


def test_extract_roi_without_padding_returns_exact_region():
    img = _grid(30, 40)
    face = Face(x=3, y=5, width=10, height=8)

    roi = face.extract_roi(img)

    np.testing.assert_array_equal(roi, img[5:13, 3:13])


def test_extract_roi_padding_expands_symmetrically():
    img = _grid(50, 50)
    face = Face(x=10, y=10, width=20, height=20)

    roi = face.extract_roi(img, padding=0.1)

    np.testing.assert_array_equal(roi, img[8:32, 8:32])
    assert roi.shape == (24, 24)


def test_extract_roi_padding_clamps_at_top_left_edge():
    img = _grid(20, 25)
    face = Face(x=0, y=0, width=10, height=10)

    roi = face.extract_roi(img, padding=0.5)

    np.testing.assert_array_equal(roi, img[0:15, 0:15])


def test_extract_roi_face_overhanging_edge_stays_in_bounds():
    img = _grid(10, 10)
    # Bounding box extends past the right and bottom edges of the image.
    face = Face(x=6, y=6, width=10, height=10)

    roi = face.extract_roi(img, padding=0.2)

    np.testing.assert_array_equal(roi, img[4:10, 4:10])


def test_extract_roi_padding_is_truncated_not_rounded():
    img = _grid(40, 40)
    face = Face(x=10, y=10, width=15, height=15)

    roi = face.extract_roi(img, padding=0.1)

    # int(15 * 0.1) == 1: each side grows by exactly one pixel.
    assert roi.shape == (17, 17)
