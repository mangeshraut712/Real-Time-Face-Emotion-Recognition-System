"""
Real-Time Face Emotion Recognition System

A state-of-the-art deep learning system for detecting and classifying
facial emotions in real-time using CNN architectures and computer vision.

Features:
- Multiple CNN architectures (EfficientNet, XCEPTION, MobileNet)
- MediaPipe and Haar cascade face detection
- Modern GUI and web interfaces
- Real-time video processing

Updated December 2024.
"""

__version__ = "2.0.0"
__author__ = "Mangesh Raut"
__license__ = "MIT"

from src.core.emotion_detector import EmotionDetector, EmotionRecognitionPipeline
from src.core.face_detector import FaceDetector, DetectorBackend

__all__ = [
    "EmotionDetector",
    "EmotionRecognitionPipeline", 
    "FaceDetector",
    "DetectorBackend",
    "__version__",
]
