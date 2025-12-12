"""
Emotion Detection Module.

This module provides emotion detection capabilities using trained CNN models
with support for multi-face processing and emotion smoothing.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from src.core.face_detector import Face, FaceDetector

logger = logging.getLogger(__name__)

# Emotion labels
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Emotion colors (BGR for OpenCV)
EMOTION_COLORS = {
    "angry": (60, 76, 231),       # Red
    "disgust": (182, 89, 155),    # Purple
    "scared": (15, 196, 241),     # Yellow
    "happy": (113, 204, 46),      # Green
    "sad": (219, 152, 52),        # Blue
    "surprised": (34, 126, 230),  # Orange
    "neutral": (166, 165, 149),   # Gray
}


@dataclass
class EmotionResult:
    """Result of emotion detection for a single face."""
    
    face: Face
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    
    @property
    def color(self) -> Tuple[int, int, int]:
        """Get the color associated with this emotion (BGR)."""
        return EMOTION_COLORS.get(self.emotion, (255, 255, 255))
    
    @property
    def emoji(self) -> str:
        """Get the emoji for this emotion."""
        emojis = {
            "angry": "😠",
            "disgust": "🤢",
            "scared": "😨",
            "happy": "😊",
            "sad": "😢",
            "surprised": "😲",
            "neutral": "😐",
        }
        return emojis.get(self.emotion, "🤔")
    
    def as_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "face": self.face.bbox,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


@dataclass
class EmotionHistory:
    """Track emotion history for smoothing predictions."""
    
    max_length: int = 10
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def add(self, probabilities: np.ndarray) -> None:
        """Add a new prediction to history."""
        self.history.append(probabilities)
    
    def get_smoothed(self) -> np.ndarray:
        """Get smoothed probabilities using exponential moving average."""
        if not self.history:
            return np.zeros(len(EMOTIONS))
        
        # Simple average
        return np.mean(list(self.history), axis=0)
    
    def clear(self) -> None:
        """Clear the history."""
        self.history.clear()


class EmotionDetector:
    """
    Emotion detector using trained CNN models.
    
    Detects facial expressions from extracted face regions and classifies
    them into one of seven emotion categories.
    
    Attributes:
        model: Loaded Keras model
        target_size: Expected input size for the model
        smooth_predictions: Whether to smooth predictions over time
    """
    
    DEFAULT_MODEL = "_mini_XCEPTION.102-0.66.hdf5"
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        target_size: Tuple[int, int] = (64, 64),
        smooth_predictions: bool = True,
        smooth_factor: float = 0.3,
    ):
        """
        Initialize the emotion detector.
        
        Args:
            model_path: Path to the trained model file
            target_size: Size to resize faces before prediction
            smooth_predictions: Whether to smooth predictions over time
            smooth_factor: Smoothing factor (0 = no smoothing, 1 = full smoothing)
        """
        self.target_size = target_size
        self.smooth_predictions = smooth_predictions
        self.smooth_factor = smooth_factor
        
        # Load model
        if model_path is None:
            model_path = self._find_default_model()
        
        self.model_path = Path(model_path)
        self._ensure_model_exists()
        
        logger.info(f"Loading emotion model from {self.model_path}")
        self.model = load_model(str(self.model_path), compile=False)
        
        # Get actual input size from model
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        target = input_shape[1:3]
        if target[0] is not None and target[1] is not None:
            self.target_size = (int(target[0]), int(target[1]))
        
        # History for smoothing (per-face tracking)
        self._histories: Dict[int, EmotionHistory] = {}
        self._last_probs: Optional[np.ndarray] = None
        
        logger.info(f"Emotion detector initialized. Target size: {self.target_size}")
    
    def _find_default_model(self) -> Path:
        """Find the default emotion model file."""
        locations = [
            Path(__file__).parent.parent.parent / "models" / self.DEFAULT_MODEL,
            Path("models") / self.DEFAULT_MODEL,
        ]
        
        for loc in locations:
            if loc.exists():
                return loc
        
        # Try to extract from zip
        zip_path = Path("models.zip")
        if zip_path.exists():
            self._extract_model_zip(zip_path)
            return locations[0]
        
        raise FileNotFoundError(
            f"Could not find emotion model. Searched: {[str(l) for l in locations]}"
        )
    
    def _extract_model_zip(self, zip_path: Path) -> None:
        """Extract model files from zip."""
        root = zip_path.parent.resolve()
        
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                name = member.filename
                if name.startswith("__MACOSX") or name.endswith(".DS_Store"):
                    continue
                if not name.startswith("models/"):
                    continue
                    
                dest = (root / name).resolve()
                if str(dest).startswith(str(root)):
                    zf.extract(member, root)
        
        logger.info(f"Extracted model files from {zip_path}")
    
    def _ensure_model_exists(self) -> None:
        """Ensure the model file exists, extracting from zip if needed."""
        if self.model_path.exists():
            return
        
        zip_path = self.model_path.parent.parent / "models.zip"
        if not zip_path.exists():
            zip_path = Path("models.zip")
        
        if zip_path.exists():
            self._extract_model_zip(zip_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess a face ROI for prediction.
        
        Args:
            roi: Grayscale face region
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Resize to target size
        if roi.shape[:2] != self.target_size:
            roi = cv2.resize(roi, self.target_size)
        
        # Add channel dimension if needed
        if len(roi.shape) == 2:
            roi = np.expand_dims(roi, axis=-1)
        
        # Convert to float and normalize
        roi = roi.astype(np.float32) / 255.0
        roi = (roi - 0.5) * 2.0
        
        # Convert to array and add batch dimension
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        return roi
    
    def predict(
        self,
        roi: np.ndarray,
        face_id: int = 0,
    ) -> EmotionResult:
        """
        Predict emotion from a face region.
        
        Args:
            roi: Grayscale face region
            face_id: ID for tracking (used for smoothing)
            
        Returns:
            EmotionResult with prediction details
        """
        # Preprocess
        tensor = self._preprocess(roi)
        
        # Predict
        probs = self.model.predict(tensor, verbose=0)[0]
        
        # Apply smoothing if enabled
        if self.smooth_predictions:
            if face_id not in self._histories:
                self._histories[face_id] = EmotionHistory()
            
            self._histories[face_id].add(probs)
            probs = self._histories[face_id].get_smoothed()
        
        self._last_probs = probs
        
        # Get top emotion
        idx = int(np.argmax(probs))
        emotion = EMOTIONS[idx]
        confidence = float(probs[idx])
        
        # Create result
        probabilities = {e: float(p) for e, p in zip(EMOTIONS, probs)}
        
        return EmotionResult(
            face=Face(0, 0, 0, 0),  # Placeholder, will be updated by caller
            emotion=emotion,
            confidence=confidence,
            probabilities=probabilities,
        )
    
    def detect_emotions(
        self,
        image: np.ndarray,
        faces: List[Face],
    ) -> List[EmotionResult]:
        """
        Detect emotions for multiple faces.
        
        Args:
            image: Input image (BGR or grayscale)
            faces: List of detected Face objects
            
        Returns:
            List of EmotionResult objects
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        results = []
        
        for i, face in enumerate(faces):
            # Extract face region
            roi = face.extract_roi(gray)
            
            # Predict
            result = self.predict(roi, face_id=i)
            
            # Update face reference
            result.face = face
            
            results.append(result)
        
        return results
    
    def draw_results(
        self,
        image: np.ndarray,
        results: List[EmotionResult],
        draw_box: bool = True,
        draw_label: bool = True,
        draw_emoji: bool = False,
    ) -> np.ndarray:
        """
        Draw emotion detection results on an image.
        
        Args:
            image: Input image to draw on
            results: List of EmotionResult objects
            draw_box: Whether to draw face bounding box
            draw_label: Whether to draw emotion label
            draw_emoji: Whether to draw emoji (requires Unicode font)
            
        Returns:
            Image with drawn results
        """
        output = image.copy()
        
        for result in results:
            face = result.face
            color = result.color
            
            # Draw bounding box
            if draw_box:
                cv2.rectangle(
                    output,
                    (face.x, face.y),
                    (face.x + face.width, face.y + face.height),
                    color,
                    2,
                )
            
            # Draw label
            if draw_label:
                label = f"{result.emotion}: {result.confidence:.1%}"
                
                # Background for text
                (text_w, text_h), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2,
                )
                
                cv2.rectangle(
                    output,
                    (face.x, face.y - text_h - 10),
                    (face.x + text_w + 10, face.y),
                    color,
                    -1,
                )
                
                cv2.putText(
                    output,
                    label,
                    (face.x + 5, face.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
        
        return output
    
    def get_last_probabilities(self) -> Optional[Dict[str, float]]:
        """Get the last predicted probabilities."""
        if self._last_probs is None:
            return None
        return {e: float(p) for e, p in zip(EMOTIONS, self._last_probs)}
    
    def reset_history(self) -> None:
        """Reset all smoothing histories."""
        self._histories.clear()
        self._last_probs = None


class EmotionRecognitionPipeline:
    """
    Complete emotion recognition pipeline.
    
    Combines face detection and emotion recognition into a single
    easy-to-use interface.
    """
    
    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        emotion_detector: Optional[EmotionDetector] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            face_detector: Face detector instance (created if None)
            emotion_detector: Emotion detector instance (created if None)
        """
        self.face_detector = face_detector or FaceDetector()
        self.emotion_detector = emotion_detector or EmotionDetector()
    
    def process_frame(
        self,
        frame: np.ndarray,
        draw_results: bool = True,
    ) -> Tuple[np.ndarray, List[EmotionResult]]:
        """
        Process a single video frame.
        
        Args:
            frame: Input BGR frame
            draw_results: Whether to draw results on the frame
            
        Returns:
            Tuple of (processed frame, list of results)
        """
        # Detect faces
        faces = self.face_detector.detect(frame)
        
        # Detect emotions
        results = self.emotion_detector.detect_emotions(frame, faces)
        
        # Draw results
        output = frame
        if draw_results and results:
            output = self.emotion_detector.draw_results(frame, results)
        
        return output, results
    
    def process_image(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
    ) -> List[EmotionResult]:
        """
        Process a single image file.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            
        Returns:
            List of EmotionResult objects
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Process
        output, results = self.process_frame(image, draw_results=output_path is not None)
        
        # Save if requested
        if output_path:
            cv2.imwrite(str(output_path), output)
        
        return results


if __name__ == "__main__":
    # Demo: Test emotion detection pipeline
    pipeline = EmotionRecognitionPipeline()
    
    cap = cv2.VideoCapture(0)
    
    print("Emotion Detection Demo - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        output, results = pipeline.process_frame(frame)
        
        # Display info
        if results:
            info = f"{results[0].emotion} ({results[0].confidence:.1%})"
        else:
            info = "No face detected"
        
        cv2.putText(
            output,
            info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        
        cv2.imshow("Emotion Detection", output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
