#!/usr/bin/env python3
"""
Training Script for Emotion Classification Models.

Train CNN models on the FER2013 dataset for facial emotion recognition.

Usage:
    python train.py --dataset fer2013/fer2013/fer2013.csv
    python train.py --model mini_xception --epochs 100
    python train.py --model simple_cnn --image-size 48
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import MODEL_REGISTRY, get_model
from src.utils.preprocessing import load_fer2013, preprocess_input

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train emotion classification models on FER2013.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
    - mini_xception  (recommended, balanced accuracy/speed)
    - tiny_xception  (fastest, lower accuracy)
    - simple_cnn     (baseline)
    - simpler_cnn    (lightweight)
    - big_xception   (highest accuracy, slower)

Examples:
    python train.py --dataset fer2013/fer2013/fer2013.csv
    python train.py --model mini_xception --epochs 100 --batch-size 64
    python train.py --model tiny_xception --image-size 48 --learning-rate 0.0005
        """,
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("fer2013/fer2013/fer2013.csv"),
        help="Path to FER2013 CSV file",
    )

    # Model
    parser.add_argument(
        "--model",
        default="mini_xception",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to train (default: mini_xception)",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Resize faces to this square size (default: 64)",
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum training epochs (default: 200)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (default: 50)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation set size (default: 0.2)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory for model checkpoints (default: models)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for logging (auto-generated if not provided)",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU if available (default: True)",
    )

    return parser.parse_args()


def setup_gpu(use_gpu: bool) -> None:
    """Configure GPU settings."""
    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")
        logger.info("GPU disabled, using CPU only")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        except RuntimeError as e:
            logger.warning(f"GPU configuration error: {e}")
    else:
        logger.info("No GPU found, using CPU")


def create_data_generator() -> ImageDataGenerator:
    """Create image data generator with augmentation."""
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "training.log"),
        ],
    )

    # Log configuration
    logger.info("=" * 60)
    logger.info("Emotion Classification Training")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Image size: {args.image_size}x{args.image_size}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 60)

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Setup GPU
    setup_gpu(args.gpu)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment name
    experiment_name = args.experiment_name or f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}"
    experiment_dir = args.output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment directory: {experiment_dir}")

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}...")
    try:
        faces, emotions = load_fer2013(
            args.dataset,
            image_size=(args.image_size, args.image_size),
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Download FER2013 from: https://www.kaggle.com/datasets/msambare/fer2013")
        sys.exit(1)

    # Preprocess
    faces = preprocess_input(faces)
    num_classes = emotions.shape[1]

    logger.info(f"Dataset shape: {faces.shape}")
    logger.info(f"Number of classes: {num_classes}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        faces,
        emotions,
        test_size=args.validation_split,
        shuffle=True,
        stratify=emotions.argmax(axis=1),
        random_state=args.seed,
    )

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # Create model
    input_shape = (args.image_size, args.image_size, 1)
    model = get_model(args.model, input_shape, num_classes)

    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Model summary:")
    model.summary(print_fn=logger.info)

    # Data augmentation
    data_generator = create_data_generator()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            str(experiment_dir / f"{args.model}.{{epoch:02d}}-{{val_accuracy:.2f}}.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        CSVLogger(str(experiment_dir / "training_history.csv")),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=max(1, args.patience // 4),
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    if args.tensorboard:
        callbacks.append(
            TensorBoard(
                log_dir=str(experiment_dir / "tensorboard"),
                histogram_freq=1,
            )
        )

    # Train
    logger.info("Starting training...")

    steps_per_epoch = max(1, len(X_train) // args.batch_size)

    history = model.fit(
        data_generator.flow(X_train, y_train, batch_size=args.batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    final_model_path = experiment_dir / f"{args.model}_final.keras"
    model.save(str(final_model_path))
    logger.info(f"Final model saved to: {final_model_path}")

    # Print results
    best_val_acc = max(history.history["val_accuracy"])
    best_epoch = history.history["val_accuracy"].index(best_val_acc) + 1

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    logger.info(f"Model saved to: {experiment_dir}")
    logger.info("=" * 60)

    return history


if __name__ == "__main__":
    main()
