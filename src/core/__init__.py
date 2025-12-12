"""
Core detection and recognition modules.

This package provides the main functionality for face detection
and emotion recognition using deep learning.
"""

from src.core.emotion_detector import (
    EMOTIONS,
    EmotionDetector,
    EmotionRecognitionPipeline,
    EmotionResult,
)
from src.core.face_detector import (
    DetectorBackend,
    Face,
    FaceDetector,
)
from src.core.models import (
    MODEL_REGISTRY,
    get_model,
    get_model_info,
    list_models,
)

__all__ = [
    # Emotion detection
    "EmotionDetector",
    "EmotionRecognitionPipeline",
    "EmotionResult",
    "EMOTIONS",
    # Face detection
    "Face",
    "FaceDetector",
    "DetectorBackend",
    # Models
    "get_model",
    "list_models",
    "get_model_info",
    "MODEL_REGISTRY",
]
