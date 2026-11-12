"""Helpers for FER2013 emotion labels."""

from __future__ import annotations

from typing import cast

import numpy as np

NUM_EMOTIONS = 7


def encode_emotion_labels(labels: np.ndarray, num_classes: int = NUM_EMOTIONS) -> np.ndarray:
    """Return an (N, num_classes) one-hot matrix for integer labels in ``0..num_classes-1``.

    ``pandas.get_dummies`` only emits columns for classes present in the
    input, so a subset of FER2013 would silently shrink the last dimension.
    Out-of-range values raise ``ValueError``.
    """
    values = np.asarray(labels)
    if values.size == 0:
        return np.zeros((0, num_classes), dtype=np.float32)
    if num_classes < 1:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("emotion labels must be numeric")
    if np.isnan(values.astype(np.float64)).any():
        raise ValueError("emotion labels must not contain NaN")
    if np.any(values != np.floor(values)):
        raise ValueError("emotion labels must be integers")
    integers = values.astype(int, copy=False)
    if np.any((integers < 0) | (integers >= num_classes)):
        raise ValueError(
            f"emotion labels must be in 0..{num_classes - 1}, got range "
            f"{int(integers.min())}..{int(integers.max())}"
        )
    return cast("np.ndarray", np.eye(num_classes, dtype=np.float32)[integers])
