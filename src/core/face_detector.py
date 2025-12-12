"""
Face Detection Module - Multi-Backend Support.

This module provides face detection capabilities with multiple backends:
- MediaPipe (default, most accurate)
- Haar Cascades (fallback, lightweight)

Updated December 2024 with MediaPipe Face Detection.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try to import MediaPipe
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.info("MediaPipe not available, using Haar cascades")


class DetectorBackend(Enum):
    """Available face detection backends."""

    MEDIAPIPE = "mediapipe"
    HAAR = "haar"
    AUTO = "auto"


@dataclass
class Face:
    """Represents a detected face with its bounding box."""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    landmarks: dict | None = None  # Face landmarks if available

    @property
    def area(self) -> int:
        """Calculate the area of the face bounding box."""
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        """Get the center point of the face."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box as (x, y, w, h) tuple."""
        return (self.x, self.y, self.width, self.height)

    @property
    def rect(self) -> tuple[int, int, int, int]:
        """Get rectangle as (x1, y1, x2, y2) tuple."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def extract_roi(
        self,
        image: np.ndarray,
        padding: float = 0.0,
    ) -> np.ndarray:
        """
        Extract the region of interest from an image.

        Args:
            image: Source image (grayscale or color)
            padding: Padding factor (0.0 = no padding, 0.1 = 10% padding)

        Returns:
            Cropped face region
        """
        h, w = image.shape[:2]

        # Calculate padding
        pad_x = int(self.width * padding)
        pad_y = int(self.height * padding)

        # Apply padding with bounds checking
        x1 = max(0, self.x - pad_x)
        y1 = max(0, self.y - pad_y)
        x2 = min(w, self.x + self.width + pad_x)
        y2 = min(h, self.y + self.height + pad_y)

        return image[y1:y2, x1:x2]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
        }


class MediaPipeFaceDetector:
    """
    Face detector using MediaPipe Face Detection.

    More accurate than Haar cascades, especially for varied poses.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,  # 0 = short range (2m), 1 = full range (5m)
    ):
        """
        Initialize MediaPipe face detector.

        Args:
            min_detection_confidence: Minimum confidence threshold
            model_selection: 0 for short-range, 1 for full-range
        """
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection,
        )
        logger.info("MediaPipe face detector initialized")

    def detect(
        self,
        image: np.ndarray,
        max_faces: int = 5,
    ) -> list[Face]:
        """
        Detect faces in an image.

        Args:
            image: Input BGR image
            max_faces: Maximum number of faces to return

        Returns:
            List of Face objects
        """
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process
        results = self.detector.process(rgb)

        if not results.detections:
            return []

        h, w = image.shape[:2]
        faces = []

        for detection in results.detections[:max_faces]:
            bbox = detection.location_data.relative_bounding_box

            # Convert relative coordinates to absolute
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Clamp to image bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            faces.append(
                Face(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=detection.score[0],
                )
            )

        # Sort by area (largest first)
        faces.sort(key=lambda f: f.area, reverse=True)

        return faces

    def close(self):
        """Release resources."""
        self.detector.close()


class HaarCascadeDetector:
    """
    Face detector using Haar cascade classifiers.

    Lightweight fallback when MediaPipe is not available.
    """

    DEFAULT_CASCADE = "haarcascade_frontalface_default.xml"

    def __init__(
        self,
        cascade_path: Path | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
    ):
        """
        Initialize Haar cascade detector.

        Args:
            cascade_path: Path to cascade XML file
            scale_factor: Scale factor for multi-scale detection
            min_neighbors: Minimum neighbors for validation
            min_size: Minimum face size
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        if cascade_path is None:
            cascade_path = self._find_cascade()

        self.cascade_path = Path(cascade_path)
        self._ensure_cascade_exists()

        self.classifier = cv2.CascadeClassifier(str(self.cascade_path))

        if self.classifier.empty():
            raise ValueError(f"Failed to load cascade from {self.cascade_path}")

        logger.info(f"Haar cascade detector initialized: {self.cascade_path.name}")

    def _find_cascade(self) -> Path:
        """Find cascade file."""
        locations = [
            Path(__file__).parent.parent.parent / "haarcascade_files" / self.DEFAULT_CASCADE,
            Path("haarcascade_files") / self.DEFAULT_CASCADE,
            Path(cv2.data.haarcascades) / self.DEFAULT_CASCADE,
        ]

        for loc in locations:
            if loc.exists():
                return loc

        # Try extracting from zip
        zip_path = Path("haarcascade_files.zip")
        if zip_path.exists():
            self._extract_zip(zip_path)
            return locations[0]

        raise FileNotFoundError(f"Cascade not found: {locations}")

    def _extract_zip(self, zip_path: Path) -> None:
        """Extract cascade from zip."""
        root = zip_path.parent.resolve()

        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                name = member.filename
                if name.startswith("__MACOSX") or ".DS_Store" in name:
                    continue
                if "haarcascade" in name.lower():
                    zf.extract(member, root)

    def _ensure_cascade_exists(self) -> None:
        """Ensure cascade exists."""
        if not self.cascade_path.exists():
            zip_path = Path("haarcascade_files.zip")
            if zip_path.exists():
                self._extract_zip(zip_path)

    def detect(
        self,
        image: np.ndarray,
        max_faces: int = 5,
    ) -> list[Face]:
        """
        Detect faces in an image.

        Args:
            image: Input image (BGR or grayscale)
            max_faces: Maximum faces to return

        Returns:
            List of Face objects
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect
        faces_raw = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        faces = [
            Face(x=int(x), y=int(y), width=int(w), height=int(h)) for (x, y, w, h) in faces_raw
        ]

        faces.sort(key=lambda f: f.area, reverse=True)

        return faces[:max_faces]

    def close(self):
        """Release resources."""


class FaceDetector:
    """
    Unified face detector with automatic backend selection.

    Supports MediaPipe (preferred) and Haar cascades (fallback).

    Attributes:
        backend: Detection backend being used
        max_faces: Maximum faces to detect
    """

    def __init__(
        self,
        backend: DetectorBackend = DetectorBackend.AUTO,
        cascade_path: Path | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
        max_faces: int = 5,
        min_confidence: float = 0.5,
    ):
        """
        Initialize the face detector.

        Args:
            backend: Detection backend (AUTO, MEDIAPIPE, or HAAR)
            cascade_path: Path to Haar cascade (for HAAR backend)
            scale_factor: Haar cascade scale factor
            min_neighbors: Haar cascade min neighbors
            min_size: Minimum face size
            max_faces: Maximum faces to return
            min_confidence: MediaPipe confidence threshold
        """
        self.max_faces = max_faces

        # Select backend
        if backend == DetectorBackend.AUTO:
            backend = DetectorBackend.MEDIAPIPE if MEDIAPIPE_AVAILABLE else DetectorBackend.HAAR

        self.backend = backend

        if backend == DetectorBackend.MEDIAPIPE:
            if not MEDIAPIPE_AVAILABLE:
                raise ImportError("MediaPipe not installed. Install with: pip install mediapipe")
            self._detector = MediaPipeFaceDetector(min_detection_confidence=min_confidence)
        else:
            self._detector = HaarCascadeDetector(
                cascade_path=cascade_path,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
                min_size=min_size,
            )

        logger.info(f"Face detector using backend: {backend.value}")

    def detect(
        self,
        image: np.ndarray,
        return_largest: bool = False,
    ) -> list[Face]:
        """
        Detect faces in an image.

        Args:
            image: Input image (BGR format)
            return_largest: If True, return only the largest face

        Returns:
            List of Face objects, sorted by size (largest first)
        """
        faces = self._detector.detect(image, max_faces=self.max_faces)

        if return_largest and faces:
            return [faces[0]]

        return faces

    def detect_and_extract(
        self,
        image: np.ndarray,
        target_size: tuple[int, int] = (64, 64),
        return_largest: bool = True,
    ) -> list[tuple[Face, np.ndarray]]:
        """
        Detect faces and extract resized ROIs.

        Args:
            image: Input BGR image
            target_size: Size to resize faces
            return_largest: Return only largest face

        Returns:
            List of (Face, roi) tuples
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        faces = self.detect(image, return_largest=return_largest)

        results = []
        for face in faces:
            roi = face.extract_roi(gray)
            roi_resized = cv2.resize(roi, target_size)
            results.append((face, roi_resized))

        return results

    def draw_faces(
        self,
        image: np.ndarray,
        faces: list[Face],
        color: tuple[int, int, int] | None = None,
        thickness: int = 2,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw face bounding boxes on an image.

        Args:
            image: Input image
            faces: List of faces to draw
            color: Box color (BGR), defaults to green
            thickness: Line thickness
            show_confidence: Show confidence score

        Returns:
            Image with drawn faces
        """
        result = image.copy()
        color = color or (0, 255, 0)

        for face in faces:
            cv2.rectangle(
                result,
                (face.x, face.y),
                (face.x + face.width, face.y + face.height),
                color,
                thickness,
            )

            if show_confidence and face.confidence < 1.0:
                label = f"{face.confidence:.0%}"
                cv2.putText(
                    result,
                    label,
                    (face.x, face.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        return result

    def close(self):
        """Release detector resources."""
        self._detector.close()

    def __enter__(self) -> FaceDetector:
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Demo
    print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")

    detector = FaceDetector()
    print(f"Using backend: {detector.backend.value}")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)
        frame = detector.draw_faces(frame, faces)

        cv2.putText(
            frame,
            f"Faces: {len(faces)} | Backend: {detector.backend.value}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
