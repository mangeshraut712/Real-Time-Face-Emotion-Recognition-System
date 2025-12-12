"""
Video Capture and Recording Utilities.

This module provides enhanced video capture and recording capabilities
with thread-safe operations and performance optimizations.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class VideoCapture:
    """
    Enhanced video capture with threading support.

    Provides thread-safe video capture with frame buffering
    for improved performance.
    """

    def __init__(
        self,
        source: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 2,
    ):
        """
        Initialize video capture.

        Args:
            source: Camera index or video file path
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
            buffer_size: Frame buffer size
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: cv2.VideoCapture | None = None
        self._frame_queue: Queue = Queue(maxsize=buffer_size)
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_frame: np.ndarray | None = None
        self._lock = threading.Lock()

        # Statistics
        self._frame_count = 0
        self._start_time: float | None = None

    def start(self) -> VideoCapture:
        """Start video capture in a background thread."""
        if self._running:
            return self

        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        # Set properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Get actual properties
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self._cap.get(cv2.CAP_PROP_FPS)) or 30

        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info(f"Video capture started: {self.width}x{self.height}@{self.fps}fps")

        return self

    def _capture_loop(self) -> None:
        """Background thread for frame capture."""
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()

            if not ret:
                if isinstance(self.source, str):
                    # Video file ended
                    break
                continue

            with self._lock:
                self._last_frame = frame
                self._frame_count += 1

            # Non-blocking put
            try:
                if self._frame_queue.full():
                    self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(frame)
            except Exception:
                pass

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the latest frame.

        Returns:
            Tuple of (success, frame)
        """
        with self._lock:
            if self._last_frame is None:
                return False, None
            return True, self._last_frame.copy()

    def read_queue(self, timeout: float = 0.1) -> tuple[bool, np.ndarray | None]:
        """
        Read a frame from the queue.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Tuple of (success, frame)
        """
        try:
            frame = self._frame_queue.get(timeout=timeout)
            return True, frame
        except Empty:
            return False, None

    def stop(self) -> None:
        """Stop video capture."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.info("Video capture stopped")

    @property
    def is_opened(self) -> bool:
        """Check if capture is opened."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def actual_fps(self) -> float:
        """Get actual frames per second."""
        if self._start_time is None or self._frame_count == 0:
            return 0.0

        elapsed = time.time() - self._start_time
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    def __enter__(self) -> VideoCapture:
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()


class VideoWriter:
    """
    Video writer with automatic codec selection and timestamp overlay.
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: int = 30,
        codec: str = "mp4v",
        add_timestamp: bool = False,
    ):
        """
        Initialize video writer.

        Args:
            output_path: Output file path
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: FourCC codec code
            add_timestamp: Whether to add timestamp overlay
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.add_timestamp = add_timestamp

        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0
        self._start_time: datetime | None = None

    def start(self) -> VideoWriter:
        """Start video recording."""
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {self.output_path}")

        self._start_time = datetime.now()
        logger.info(f"Recording started: {self.output_path}")

        return self

    def write(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.

        Args:
            frame: Frame to write (BGR format)
        """
        if self._writer is None:
            return

        # Add timestamp if enabled
        if self.add_timestamp:
            frame = frame.copy()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                frame,
                timestamp,
                (10, self.height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        self._writer.write(frame)
        self._frame_count += 1

    def stop(self) -> None:
        """Stop video recording."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

            logger.info(
                f"Recording stopped: {self._frame_count} frames written to {self.output_path}"
            )

    @property
    def duration(self) -> float:
        """Get recording duration in seconds."""
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    def __enter__(self) -> VideoWriter:
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()


def capture_screenshot(
    frame: np.ndarray,
    output_dir: Path = Path("screenshots"),
    prefix: str = "emotion_capture",
) -> Path:
    """
    Save a screenshot of the current frame.

    Args:
        frame: Frame to save
        output_dir: Directory to save screenshots
        prefix: Filename prefix

    Returns:
        Path to saved screenshot
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    output_path = output_dir / filename

    cv2.imwrite(str(output_path), frame)
    logger.info(f"Screenshot saved: {output_path}")

    return output_path
