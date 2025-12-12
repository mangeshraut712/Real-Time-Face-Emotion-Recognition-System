"""
Core detection and recognition modules.

This package provides the main functionality for face detection
and emotion recognition using deep learning.
"""

from src.core.emotion_detector import (
    EmotionDetector,
    EmotionRecognitionPipeline,
    EmotionResult,
    EMOTIONS,
)
from src.core.face_detector import (
    Face,
    FaceDetector,
    DetectorBackend,
)
from src.core.models import (
    get_model,
    list_models,
    get_model_info,
    MODEL_REGISTRY,
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
