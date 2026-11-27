"""Unit tests for src.utils.video.VideoCapture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.utils.video as video_module
from src.utils.video import VideoCapture


class _FakeCapture:
    """Minimal stand-in for cv2.VideoCapture serving a finite frame list."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_args):
        return True

    def get(self, _prop):
        return 0

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        self.released = True


def test_read_returns_defensive_copy_of_last_frame():
    cap = VideoCapture()
    cap._last_frame = np.zeros((4, 4, 3), dtype=np.uint8)

    ok, frame = cap.read()

    assert ok
    frame[:] = 255
    np.testing.assert_array_equal(cap._last_frame, np.zeros((4, 4, 3), dtype=np.uint8))


def test_read_before_first_frame_reports_failure():
    cap = VideoCapture()

    ok, frame = cap.read()

    assert ok is False
    assert frame is None


def test_read_queue_empty_times_out():
    cap = VideoCapture()

    ok, frame = cap.read_queue(timeout=0.01)

    assert ok is False
    assert frame is None


def test_read_queue_returns_buffered_frame():
    cap = VideoCapture()
    frame = np.full((2, 2), 7, dtype=np.uint8)
    cap._frame_queue.put_nowait(frame)

    ok, out = cap.read_queue(timeout=0.01)

    assert ok
    np.testing.assert_array_equal(out, frame)


def test_start_falls_back_to_30fps_when_device_reports_zero(monkeypatch):
    fake = _FakeCapture(frames=[])
    monkeypatch.setattr(video_module.cv2, "VideoCapture", lambda *_: fake)

    cap = VideoCapture(source=0)
    cap.start()
    try:
        assert cap.fps == 30
    finally:
        cap.stop()


def test_capture_thread_exits_when_path_video_ends(monkeypatch):
    """Regression: a Path video source must behave like a file source so the
    capture loop terminates at end-of-file instead of busy-looping forever.
    """
    fake = _FakeCapture(frames=[np.zeros((2, 2, 3), dtype=np.uint8)])
    monkeypatch.setattr(video_module.cv2, "VideoCapture", lambda *_: fake)

    cap = VideoCapture(source=Path("clip.mp4"))
    cap.start()

    assert cap._thread is not None
    cap._thread.join(timeout=2.0)
    try:
        assert not cap._thread.is_alive()
    finally:
        cap.stop()
    assert fake.released


def test_path_source_is_passed_to_opencv_as_str(monkeypatch):
    opened = []

    def fake_capture(source):
        opened.append(source)
        return _FakeCapture(frames=[])

    monkeypatch.setattr(video_module.cv2, "VideoCapture", fake_capture)

    cap = VideoCapture(source=Path("clips") / "clip.mp4")
    assert cap.source == str(Path("clips") / "clip.mp4")
    cap.start()
    cap.stop()
    assert opened == [str(Path("clips") / "clip.mp4")]
    assert VideoCapture(source=1).source == 1


@pytest.mark.parametrize(
    "source",
    [0, 1],
    ids=["default camera", "second camera"],
)
def test_camera_index_source_stays_an_int(source):
    """Regression: the capture loop tells a file from a camera with
    ``isinstance(self.source, str)`` (see ``_capture_loop``), so a camera index
    must be stored as an int. ``os.fspath`` raises TypeError on an int, which is
    why the normalisation is guarded with an ``isinstance(..., os.PathLike)``
    check instead of being applied to every source.
    """
    cap = VideoCapture(source=source) if source else VideoCapture()

    assert cap.source == source
    assert type(cap.source) is int
    assert not isinstance(cap.source, str)


def test_default_source_is_the_first_camera():
    assert VideoCapture().source == 0
