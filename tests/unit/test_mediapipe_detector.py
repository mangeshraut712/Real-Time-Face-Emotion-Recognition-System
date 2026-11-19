"""MediaPipeFaceDetector.detect keeps the largest faces, not the first ones."""

from types import SimpleNamespace

import numpy as np

from src.core.face_detector import MediaPipeFaceDetector


def _detection(xmin, ymin, size, score=0.9):
    box = SimpleNamespace(xmin=xmin, ymin=ymin, width=size, height=size)
    return SimpleNamespace(location_data=SimpleNamespace(relative_bounding_box=box), score=[score])


def test_detect_returns_largest_faces_when_over_max_faces():
    detector = MediaPipeFaceDetector.__new__(MediaPipeFaceDetector)
    detections = [_detection(0.0, 0.0, 0.1), _detection(0.2, 0.2, 0.15), _detection(0.5, 0.5, 0.4)]
    detector.detector = SimpleNamespace(process=lambda _rgb: SimpleNamespace(detections=detections))

    faces = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), max_faces=2)

    assert [face.width for face in faces] == [40, 15]
