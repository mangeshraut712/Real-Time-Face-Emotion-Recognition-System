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


def test_detect_clips_boxes_that_cross_the_left_or_top_edge():
    # MediaPipe returns negative xmin/ymin for faces touching the frame border.
    # The box must be cut to the visible part, not shifted right/down.
    detector = MediaPipeFaceDetector.__new__(MediaPipeFaceDetector)
    detections = [_detection(-0.1, 0.2, 0.3), _detection(0.5, -0.05, 0.3)]
    detector.detector = SimpleNamespace(process=lambda _rgb: SimpleNamespace(detections=detections))

    faces = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), max_faces=2)

    boxes = sorted((face.x, face.y, face.width, face.height) for face in faces)
    assert boxes == [(0, 20, 20, 30), (50, 0, 30, 25)]


def test_detect_drops_boxes_with_no_visible_pixels():
    detector = MediaPipeFaceDetector.__new__(MediaPipeFaceDetector)
    # Entirely left of the frame, and a sliver that rounds to zero width.
    detections = [_detection(-0.5, 0.2, 0.3), _detection(-0.304, 0.2, 0.3), _detection(0.4, 0.4, 0.2)]
    detector.detector = SimpleNamespace(process=lambda _rgb: SimpleNamespace(detections=detections))

    faces = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), max_faces=5)

    assert [(face.x, face.y, face.width, face.height) for face in faces] == [(40, 40, 20, 20)]
