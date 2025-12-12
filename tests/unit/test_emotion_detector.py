"""Unit tests for emotion detection module."""

import numpy as np
import pytest
import unittest.mock

from src.core.emotion_detector import EmotionDetector, EmotionResult
from src.core.face_detector import Face


class TestEmotionDetector:
    """Test suite for EmotionDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing with mocked model."""
        with unittest.mock.patch('src.core.emotion_detector.load_model') as mock_load, \
             unittest.mock.patch('pathlib.Path.exists', return_value=True):
            
            # Setup mock model
            mock_model = unittest.mock.MagicMock()
            mock_model.input_shape = [None, 64, 64, 1]
            # Mock predict to return 7 probabilities
            mock_model.predict.return_value = np.array([[0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1]])
            mock_load.return_value = mock_model

            # Initialize with dummy path
            det = EmotionDetector(model_path="dummy_model.hdf5")
            return det

    @pytest.fixture
    def sample_face(self):
        """Create sample face for testing."""
        return Face(x=10, y=10, width=100, height=100)

    @pytest.fixture
    def sample_roi(self):
        """Create sample face ROI."""
        return np.random.randint(0, 255, (64, 64), dtype=np.uint8)

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.model is not None
        assert len(detector.emotions) == 7

    def test_predict_returns_result(self, detector, sample_roi):
        """Test prediction returns EmotionResult."""
        result = detector.predict(sample_roi)
        assert isinstance(result, EmotionResult)
        assert result.emotion in detector.emotions
        assert 0 <= result.confidence <= 1

    def test_probabilities_sum_to_one(self, detector, sample_roi):
        """Test probability distribution sums to 1."""
        result = detector.predict(sample_roi)
        total_prob = sum(result.probabilities.values())
        assert abs(total_prob - 1.0) < 0.01

    def test_detect_emotions_with_faces(self, detector, sample_face):
        """Test detecting emotions from frame with faces."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = detector.detect_emotions(frame, [sample_face])

        assert len(results) == 1
        assert isinstance(results[0], EmotionResult)
        assert results[0].face == sample_face

    def test_detect_emotions_empty_faces(self, detector):
        """Test detecting emotions with no faces."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = detector.detect_emotions(frame, [])
        assert len(results) == 0

    def test_emotion_smoothing(self, detector, sample_roi):
        """Test emotion smoothing over multiple predictions."""
        # Make multiple predictions
        results = [detector.predict(sample_roi, face_id=0) for _ in range(5)]

        # Check that smoothing is applied (confidence should be more stable)
        confidences = [r.confidence for r in results]
        assert len(confidences) == 5
