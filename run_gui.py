#!/usr/bin/env python3
"""
Real-Time Emotion Recognition GUI Application.

Launch the modern tkinter-based GUI for real-time facial emotion detection.

Usage:
    python run_gui.py
    python run_gui.py --camera 1
    python run_gui.py --emotion-model models/custom_model.keras
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.gui.app import run_app


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time face emotion recognition GUI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_gui.py                          # Use default camera (0)
    python run_gui.py --camera 1               # Use camera index 1
    python run_gui.py --emotion-model custom.keras  # Use custom model
        """,
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for OpenCV (default: 0)",
    )

    parser.add_argument(
        "--detection-model",
        type=Path,
        default=None,
        help="Path to Haar cascade XML for face detection",
    )

    parser.add_argument(
        "--emotion-model",
        type=Path,
        default=None,
        help="Path to trained emotion model (.keras or .hdf5)",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Emotion Recognition GUI...")

    # Ensure assets are extracted

    try:
        run_app(
            camera_index=args.camera,
            detection_model_path=args.detection_model,
            emotion_model_path=args.emotion_model,
        )
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
