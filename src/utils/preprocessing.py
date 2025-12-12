"""
Data Preprocessing Utilities.

This module provides functions for loading and preprocessing image data
for training and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd


def preprocess_input(
    x: np.ndarray,
    v2: bool = True,
) -> np.ndarray:
    """
    Normalize input images for model inference.
    
    Applies normalization to match the preprocessing used during training.
    
    Args:
        x: Input array of images
        v2: If True, normalize to [-1, 1], else to [0, 1]
        
    Returns:
        Normalized image array
    """
    x = x.astype("float32") / 255.0
    if v2:
        x = (x - 0.5) * 2.0
    return x


def load_fer2013(
    dataset_path: Union[str, Path],
    image_size: Tuple[int, int] = (48, 48),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the FER2013 dataset.
    
    Reads the FER2013 CSV file and converts pixel strings to image arrays.
    
    Args:
        dataset_path: Path to fer2013.csv
        image_size: Target size for resizing images
        
    Returns:
        Tuple of (faces, emotions) where:
            - faces: Array of shape (N, H, W, 1) with pixel values
            - emotions: One-hot encoded emotion labels (N, 7)
            
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset format is invalid
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"FER2013 CSV not found at {dataset_path}. "
            "Download fer2013.csv from Kaggle and place it in the expected location."
        )
    
    # Load CSV
    print(f"Loading FER2013 from {dataset_path}...")
    data = pd.read_csv(dataset_path)
    
    # Validate columns
    required_columns = {"pixels", "emotion"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"Invalid dataset format. Required columns: {required_columns}, "
            f"Found: {set(data.columns)}"
        )
    
    pixels = data["pixels"].tolist()
    original_size = (48, 48)  # FER2013 original size
    
    # Process images
    print(f"Processing {len(pixels)} images...")
    faces = np.empty((len(pixels), image_size[0], image_size[1]), dtype=np.float32)
    
    for i, pixel_sequence in enumerate(pixels):
        # Convert pixel string to array
        face = np.fromstring(pixel_sequence, dtype=np.float32, sep=" ")
        face = face.reshape(original_size)
        
        # Resize if needed
        if image_size != original_size:
            face = cv2.resize(face.astype("uint8"), image_size)
        
        faces[i] = face
        
        # Progress indicator
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(pixels)} images")
    
    # Add channel dimension
    faces = np.expand_dims(faces, -1)
    
    # One-hot encode emotions
    emotions = pd.get_dummies(data["emotion"]).to_numpy(dtype=np.float32)
    
    print(f"Loaded {len(faces)} images with shape {faces.shape}")
    print(f"Emotion distribution: {dict(data['emotion'].value_counts())}")
    
    return faces, emotions


def augment_image(
    image: np.ndarray,
    rotation_range: int = 10,
    width_shift_range: float = 0.1,
    height_shift_range: float = 0.1,
    zoom_range: float = 0.1,
    horizontal_flip: bool = True,
) -> np.ndarray:
    """
    Apply random augmentation to an image.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        rotation_range: Max rotation in degrees
        width_shift_range: Max horizontal shift as fraction
        height_shift_range: Max vertical shift as fraction
        zoom_range: Max zoom factor
        horizontal_flip: Whether to randomly flip horizontally
        
    Returns:
        Augmented image
    """
    h, w = image.shape[:2]
    
    # Random rotation
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Random shifts
    if width_shift_range > 0 or height_shift_range > 0:
        tx = np.random.uniform(-width_shift_range, width_shift_range) * w
        ty = np.random.uniform(-height_shift_range, height_shift_range) * h
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Random zoom
    if zoom_range > 0:
        zoom = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, zoom)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Horizontal flip
    if horizontal_flip and np.random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    return image


def normalize_face(
    face: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """
    Normalize a face image for inference.
    
    Args:
        face: Input face image (grayscale)
        target_size: Target size for resizing
        
    Returns:
        Normalized face ready for model input
    """
    # Ensure grayscale
    if len(face.shape) == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Resize
    face = cv2.resize(face, target_size)
    
    # Histogram equalization for lighting normalization
    face = cv2.equalizeHist(face.astype(np.uint8))
    
    # Add channel dimension
    face = np.expand_dims(face, axis=-1)
    
    # Normalize to [-1, 1]
    face = preprocess_input(face)
    
    return face
